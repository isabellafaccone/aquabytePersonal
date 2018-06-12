function train()
	function mul32(a,b)
		return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
	end

	function make_patch(src, dst, dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
		local m = {1, 0, -dim4, 0, 1, -dim3}
		m = mul32({1, 0, trans[1], 0, 1, trans[2]}, m) -- translate
		m = mul32({scale[1], 0, 0, 0, scale[2], 0}, m) -- scale
		local c = math.cos(phi)
		local s = math.sin(phi)
		m = mul32({c, s, 0, -s, c, 0}, m) -- rotate
		m = mul32({1, hshear, 0, 0, 1, 0}, m) -- shear
		m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
		m = torch.FloatTensor(m)
		cv.warp_affine(src, dst, m)
		dst:mul(contrast):add(brightness)
	end

	-- subset training dataset
	if opt.subset < 1 then
		function sample(xs, p)
			local perm = torch.randperm(xs:nElement()):long()
			return xs:index(1, perm[{{1, xs:size(1) * p}}])
		end

		local tr_subset
		if dataset == 'kitti' or dataset == 'kitti2015' then
			tr_subset = sample(tr, opt.subset)
		elseif dataset == 'mb' then
			tr_2014 = sample(torch.range(11, 23):long(), opt.subset)
			tr_2006 = sample(torch.range(24, 44):long(), opt.subset)
			tr_2005 = sample(torch.range(45, 50):long(), opt.subset)
			tr_2003 = sample(torch.range(51, 52):long(), opt.subset)
			tr_2001 = sample(torch.range(53, 60):long(), opt.subset)

			tr_subset = torch.cat(tr_2014, tr_2006)
			tr_subset = torch.cat(tr_subset, tr_2005)
			tr_subset = torch.cat(tr_subset, tr_2003)
			tr_subset = torch.cat(tr_subset, tr_2001)
		end

		local nnz_tr_output = torch.FloatTensor(nnz_tr:size()):zero()
		local t = adcensus.subset_dataset(tr_subset, nnz_tr, nnz_tr_output);
		nnz_tr = nnz_tr_output[{{1,t}}]
	end

	collectgarbage()

	if opt.a == 'train_all' then
		nnz = torch.cat(nnz_tr, nnz_te, 1)
	elseif opt.a == 'train_tr' or opt.a == 'time' then
		nnz = nnz_tr
	end

	if opt.a ~= 'time' then
		perm = torch.randperm(nnz:size(1))
	end

	local fm = torch.totable(torch.linspace(opt.fm, opt.fm, opt.l1):int())

	-- network for training
	if arch == 'slow' then
		net_tr = nn.Sequential()
		for i = 1,#fm do
			net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
			net_tr:add(cudnn.ReLU(true))
		end
		net_tr:add(nn.Reshape(opt.bs, 2 * fm[#fm]))
		for i = 1,opt.l2 do
			net_tr:add(nn.Linear(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
			net_tr:add(cudnn.ReLU(true))
		end
		net_tr:add(nn.Linear(opt.nh2, 1))
		net_tr:add(cudnn.Sigmoid(false))
		net_tr:cuda()
		criterion = nn.BCECriterion2():cuda()

		-- network for testing (make sure it's synched with net_tr)
		local pad = (opt.ks - 1) / 2
		net_te = nn.Sequential()
		for i = 1,#fm do
			net_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
			net_te:add(cudnn.ReLU(true))
		end
		net_te:cuda()

		net_te2 = nn.Sequential()
		for i = 1,opt.l2 do
			net_te2:add(nn.SpatialConvolution1_fw(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
			net_te2:add(cudnn.ReLU(true))
		end
		net_te2:add(nn.SpatialConvolution1_fw(opt.nh2, 1))
		net_te2:add(cudnn.Sigmoid(true))
		net_te2:cuda()

		-- tie weights
		net_te_all = {}
		for i, v in ipairs(net_te.modules) do table.insert(net_te_all, v) end
		for i, v in ipairs(net_te2.modules) do table.insert(net_te_all, v) end

		local finput = torch.CudaTensor()
		local i_tr = 1
		local i_te = 1
		while i_tr <= net_tr:size() do
			local module_tr = net_tr:get(i_tr)
			local module_te = net_te_all[i_te]

			local skip = {['nn.Reshape']=1, ['nn.Dropout']=1}
			while skip[torch.typename(module_tr)] do
				i_tr = i_tr + 1
				module_tr = net_tr:get(i_tr)
			end

			if module_tr.weight then
				-- print(('tie weights of %s and %s'):format(torch.typename(module_te), torch.typename(module_tr)))
				assert(module_te.weight:nElement() == module_tr.weight:nElement())
				assert(module_te.bias:nElement() == module_tr.bias:nElement())
				module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
				module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
			end

			i_tr = i_tr + 1
			i_te = i_te + 1
		end
	elseif arch == 'fast' then
		net_tr = nn.Sequential()
		for i = 1,#fm do
			net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
			if i < #fm then
				net_tr:add(cudnn.ReLU(true))
			end
		end
		net_tr:add(nn.Normalize2())
		net_tr:add(nn.StereoJoin1())
		net_tr:cuda()

		net_te = net_tr:clone('weight', 'bias')
		net_te.modules[#net_te.modules] = nn.StereoJoin(1):cuda()
		for i = 1,#net_te.modules do
			local m = net_te:get(i)
			if torch.typename(m) == 'cudnn.SpatialConvolution' then
				m.padW = 1
				m.padH = 1
			end
		end

		criterion = nn.Margin2(opt.m, opt.pow):cuda()
	end

	print_net(net_tr)

	params = {}
	grads = {}
	momentums = {}
	for i = 1,net_tr:size() do
		local m = net_tr:get(i)
		if m.weight then
			m.weight_v = torch.CudaTensor(m.weight:size()):zero()
			table.insert(params, m.weight)
			table.insert(grads, m.gradWeight)
			table.insert(momentums, m.weight_v)
		end
		if m.bias then
			m.bias_v = torch.CudaTensor(m.bias:size()):zero()
			table.insert(params, m.bias)
			table.insert(grads, m.gradBias)
			table.insert(momentums, m.bias_v)
		end
	end

	ws = get_window_size(net_tr)
	x_batch_tr = torch.CudaTensor(opt.bs * 2, n_input_plane, ws, ws)
	y_batch_tr = torch.CudaTensor(opt.bs)
	x_batch_tr_ = torch.FloatTensor(x_batch_tr:size())
	y_batch_tr_ = torch.FloatTensor(y_batch_tr:size())

	time = sys.clock()

	for epoch = 1,14 do
		if opt.a == 'time' then
			break
		end
		if epoch == 12 then
			opt.lr = opt.lr / 10
		end

		local err_tr = 0
		local err_tr_cnt = 0
		for t = 1,nnz:size(1) - opt.bs/2,opt.bs/2 do
			for i = 1,opt.bs/2 do
				d_pos = torch.uniform(-opt.true1, opt.true1)
				d_neg = torch.uniform(opt.false1, opt.false2)
				if torch.uniform() < 0.5 then
					d_neg = -d_neg
				end

				assert(opt.hscale <= 1 and opt.scale <= 1)
				local s = torch.uniform(opt.scale, 1)
				local scale = {s * torch.uniform(opt.hscale, 1), s}
				if opt.hflip == 1 and torch.uniform() < 0.5 then
					scale[1] = -scale[1]
				end
				if opt.vflip == 1 and torch.uniform() < 0.5 then
					scale[2] = -scale[2]
				end
				local hshear = torch.uniform(-opt.hshear, opt.hshear)
				local trans = {torch.uniform(-opt.trans, opt.trans), torch.uniform(-opt.trans, opt.trans)}
				local phi = torch.uniform(-opt.rotate * math.pi / 180, opt.rotate * math.pi / 180)
				local brightness = torch.uniform(-opt.brightness, opt.brightness)
				assert(opt.contrast >= 1 and opt.d_contrast >= 1)
				local contrast = torch.uniform(1 / opt.contrast, opt.contrast)

				local scale_ = {scale[1] * torch.uniform(opt.d_hscale, 1), scale[2]}
				local hshear_ = hshear + torch.uniform(-opt.d_hshear, opt.d_hshear)
				local trans_ = {trans[1], trans[2] + torch.uniform(-opt.d_vtrans, opt.d_vtrans)}
				local phi_ = phi + torch.uniform(-opt.d_rotate * math.pi / 180, opt.d_rotate * math.pi / 180)
				local brightness_ = brightness + torch.uniform(-opt.d_brightness, opt.d_brightness)
				local contrast_ = contrast * torch.uniform(1 / opt.d_contrast, opt.d_contrast)

				local ind = perm[t + i - 1]
				img = nnz[{ind, 1}]
				dim3 = nnz[{ind, 2}]
				dim4 = nnz[{ind, 3}]
				d = nnz[{ind, 4}]
				if dataset == 'kitti' or dataset == 'kitti2015' then
					x0 = X0[img]
					x1 = X1[img]
					elseif dataset == 'mb' then
						light = (torch.random() % (#X[img] - 1)) + 2
						exp = (torch.random() % X[img][light]:size(1)) + 1
						light_ = light
						exp_ = exp
						if torch.uniform() < opt.d_exp then
							exp_ = (torch.random() % X[img][light]:size(1)) + 1
						end
						if torch.uniform() < opt.d_light then
							light_ = math.max(2, light - 1)
						end
						x0 = X[img][light][{exp,1}]
						x1 = X[img][light_][{exp_,2}]
					end

					make_patch(x0, x_batch_tr_[i * 4 - 3], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
					make_patch(x1, x_batch_tr_[i * 4 - 2], dim3, dim4 - d + d_pos, scale_, phi_, trans_, hshear_, brightness_, contrast_)
					make_patch(x0, x_batch_tr_[i * 4 - 1], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
					make_patch(x1, x_batch_tr_[i * 4 - 0], dim3, dim4 - d + d_neg, scale_, phi_, trans_, hshear_, brightness_, contrast_)

					y_batch_tr_[i * 2 - 1] = 0
					y_batch_tr_[i * 2] = 1
				end

				x_batch_tr:copy(x_batch_tr_)
				y_batch_tr:copy(y_batch_tr_)

				for i = 1,#params do
					grads[i]:zero()
				end

				net_tr:forward(x_batch_tr)
				local err = criterion:forward(net_tr.output, y_batch_tr)
				if err >= 0 and err < 100 then
					err_tr = err_tr + err
					err_tr_cnt = err_tr_cnt + 1
				else
					print(('WARNING! err=%f'):format(err))
				end

				criterion:backward(net_tr.output, y_batch_tr)
				net_tr:backward(x_batch_tr, criterion.gradInput)

				for i = 1,#params do
					momentums[i]:mul(opt.mom):add(-opt.lr, grads[i])
					params[i]:add(momentums[i])
				end
			end

			if opt.debug then
				save_net(epoch)
			end
			print(epoch, err_tr / err_tr_cnt, opt.lr, sys.clock() - time)
			collectgarbage()
		end
		opt.net_fname = save_net(0)
		if opt.a == 'train_tr' then
			opt.a = 'test_te'
		elseif opt.a == 'train_all' then
			opt.a = 'submit'
		end
		collectgarbage()
	end