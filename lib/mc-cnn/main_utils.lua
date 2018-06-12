function isnan(n)
   return tostring(n) == tostring(0/0)
end

function fromfile(fname)
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end

   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end

   x = x:reshape(torch.LongStorage(dim))
   return x
end

function get_window_size(net)
   ws = 1
   for i = 1,#net.modules do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' then
         ws = ws + module.kW - 1
      end
   end
   return ws
end

function savePNG(fname, x, isvol)
   local pred
   local pred_jet = torch.Tensor(1, 3, x:size(3), x:size(4))

   if isvol == true then
      pred = torch.CudaTensor(1, 1, x:size(3), x:size(4))
      adcensus.spatial_argmin(x, pred)
   else
      pred = x:double():add(1)
   end
   adcensus.grey2jet(pred[{1,1}]:div(disp_max):double(), pred_jet)
   image.savePNG(fname, pred_jet[1])
end

function saveOutlier(fname, x0, outlier)
   local img = torch.Tensor(1,3,height,img_width)
   img[{1,1}]:copy(x0)
   img[{1,2}]:copy(x0)
   img[{1,3}]:copy(x0)
   for i=1,height do
      for j=1,img_width do
         if outlier[{1,1,i,j}] == 1 then
            img[{1,1,i,j}] = 0
            img[{1,2,i,j}] = 1
            img[{1,3,i,j}] = 0
         elseif outlier[{1,1,i,j}] == 2 then
            img[{1,1,i,j}] = 1
            img[{1,2,i,j}] = 0
            img[{1,3,i,j}] = 0
         end
      end
   end
   image.savePNG(fname, img[1])
end

function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end

function print_net(net)
   local s
   local t = torch.typename(net) 
   if t == 'cudnn.SpatialConvolution' then
      print(('conv(in=%d, out=%d, k=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW))
   elseif t == 'nn.SpatialConvolutionMM_dsparse' then
      print(('conv_dsparse(in=%d, out=%d, k=%d, s=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW, net.sW))
   elseif t == 'cudnn.SpatialMaxPooling' then
      print(('max_pool(k=%d, d=%d)'):format(net.kW, net.dW))
   elseif t == 'nn.StereoJoin' then
      print(('StereoJoin(%d)'):format(net.disp_max))
   elseif t == 'nn.Margin2' then
      print(('Margin2(margin=%f, pow=%d)'):format(opt.m, opt.pow))
   elseif t == 'nn.GHCriterion' then
      print(('GHCriterion(m_pos=%f, m_neg=%f, pow=%d)'):format(opt.m_pos, opt.m_neg, opt.pow))
   elseif t == 'nn.Sequential' then
      for i = 1,#net.modules do
         print_net(net.modules[i])
      end
   else
      print(net)
   end
end

function clean_net(net)
   net.output = torch.CudaTensor()
   net.gradInput = nil
   net.weight_v = nil
   net.bias_v = nil
   net.gradWeight = nil
   net.gradBias = nil
   net.iDesc = nil
   net.oDesc = nil
   net.finput = torch.CudaTensor()
   net.fgradInput = torch.CudaTensor()
   net.tmp_in = torch.CudaTensor()
   net.tmp_out = torch.CudaTensor()
   if net.modules then
      for _, module in ipairs(net.modules) do
         clean_net(module)
      end
   end
   return net
end

function save_net(epoch)
   if arch == 'slow' then
      obj = {clean_net(net_te), clean_net(net_te2), opt}
   elseif arch == 'fast' then
      obj = {clean_net(net_te), opt}
   end
   if epoch == 0 then
      fname = ('net/net_%s.t7'):format(cmd_str)
   else
      fname = ('net/net_%s_%d.t7'):format(cmd_str, epoch)
   end
   torch.save(fname, obj, 'ascii')
   return fname
end

function forward_free(net, input)
   local currentOutput = input
   for i=1,#net.modules do
      net.modules[i].oDesc = nil
      local nextOutput = net.modules[i]:updateOutput(currentOutput)
      if currentOutput:storage() ~= nextOutput:storage() then
         currentOutput:storage():resize(1)
         currentOutput:resize(0)
      end
      currentOutput = nextOutput
   end
   net.output = currentOutput
   return currentOutput
end

function fix_border(net, vol, direction)
   local n = (get_window_size(net) - 1) / 2
   for i=1,n do
      vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
   end
end

function stereo_predict(video_id, frame_number, x_batch, id)
   local vols, vol

   if arch == 'ad' then
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      adcensus.ad(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
      adcensus.ad(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
   end

   if arch == 'census' then
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      adcensus.census(x_batch[{{1}}], x_batch[{{2}}], vols[{{1}}], -1)
      adcensus.census(x_batch[{{2}}], x_batch[{{1}}], vols[{{2}}], 1)
   end

   if arch == 'fast' then
      forward_free(net_te, x_batch:clone())
      --vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
      vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4))
      adcensus.StereoJoin(net_te.output[{{1}}], net_te.output[{{2}}], vols[{{1}}], vols[{{2}}])
      fix_border(net_te, vols[{{1}}], -1)
      fix_border(net_te, vols[{{2}}], 1)
      clean_net(net_te)
   end

   disp = {}
   local mb_directions = opt.a == 'predict' and {1, -1} or {-1}
   for _, direction in ipairs(dataset == 'mb' and mb_directions or {1, -1}) do
      sm_active = true

      if arch == 'slow' then
         if opt.use_cache then
            vol = torch.load(('cache/%s_%d.t7'):format(id, direction))
         else
            local output = forward_free(net_te, x_batch:clone())
            clean_net(net_te)
            collectgarbage()

            vol = torch.CudaTensor(1, disp_max, output:size(3), output:size(4)):fill(0 / 0)
            collectgarbage()
            for d = 1,disp_max do
               local l = output[{{1},{},{},{d,-1}}]
               local r = output[{{2},{},{},{1,-d}}]
               x_batch_te2:resize(2, r:size(2), r:size(3), r:size(4))
               x_batch_te2[1]:copy(l)
               x_batch_te2[2]:copy(r)
               x_batch_te2:resize(1, 2 * r:size(2), r:size(3), r:size(4))
               forward_free(net_te2, x_batch_te2)
               vol[{1,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(net_te2.output[{1,1}])
            end
            clean_net(net_te2)
            fix_border(net_te, vol, direction)
            if opt.make_cache then
               torch.save(('cache/%s_%d.t7'):format(id, direction), vol)
            end
         end
         collectgarbage()
      elseif arch == 'fast' or arch == 'ad' or arch == 'census' then
         vol = vols[{{direction == -1 and 1 or 2}}]
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cnn')

      -- cross computation
      local x0c, x1c
      if sm_active and opt.sm_skip ~= 'cbca' then
         x0c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         x1c = torch.CudaTensor(1, 4, vol:size(3), vol:size(4))
         adcensus.cross(x_batch[1], x0c, opt.L1, opt.tau1)
         adcensus.cross(x_batch[2], x1c, opt.L1, opt.tau1)
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i1 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
         tmp_cbca = nil
         collectgarbage()
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca1')

      if sm_active and opt.sm_skip ~= 'sgm' then
         vol = vol:transpose(2, 3):transpose(3, 4):clone()
         collectgarbage()
         do
            local out = torch.CudaTensor(1, vol:size(2), vol:size(3), vol:size(4))
            local tmp = torch.CudaTensor(vol:size(3), vol:size(4))
            for _ = 1,opt.sgm_i do
               out:zero()
               adcensus.sgm2(x_batch[1], x_batch[2], vol, out, tmp, opt.pi1, opt.pi2, opt.tau_so,
                  opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
               vol:copy(out):div(4)
            end
            vol:resize(1, disp_max, x_batch:size(3), x_batch:size(4))
            vol:copy(out:transpose(3, 4):transpose(2, 3)):div(4)

--            local out = torch.CudaTensor(4, vol:size(2), vol:size(3), vol:size(4))
--            out:zero()
--            adcensus.sgm3(x_batch[1], x_batch[2], vol, out, opt.pi1, opt.pi2, opt.tau_so,
--               opt.alpha1, opt.sgm_q1, opt.sgm_q2, direction)
--            vol:mean(out, 1)
--            vol = vol:transpose(3, 4):transpose(2, 3):clone()
         end
         collectgarbage()
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'sgm')

      if sm_active and opt.sm_skip ~= 'cbca' then
         local tmp_cbca = torch.CudaTensor(1, disp_max, vol:size(3), vol:size(4))
         for i = 1,opt.cbca_i2 do
            adcensus.cbca(x0c, x1c, vol, tmp_cbca, direction)
            vol:copy(tmp_cbca)
         end
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'cbca2')

      if opt.a == 'predict' then
         local fname = direction == -1 and 'left' or 'right'
         print(('Writing /videos/%i/stereo/%i-%s.bin, %d x %d x %d x %d'):format(video_id, frame_number, fname, vol:size(1), vol:size(2), vol:size(3), vol:size(4)))
         torch.DiskFile(('/videos/%i/stereo/%i-%s.bin'):format(video_id, frame_number, fname), 'w'):binary():writeFloat(vol:float():storage())
         collectgarbage()
      end

      _, d = torch.min(vol, 2)
      disp[direction == 1 and 1 or 2] = d:cuda():add(-1)
   end
   collectgarbage()

   if dataset == 'aquabyte' or dataset == 'kitti' or dataset == 'kitti2015' then
      local outlier = torch.CudaTensor():resizeAs(disp[2]):zero()
      adcensus.outlier_detection(disp[2], disp[1], outlier, disp_max)
      if sm_active and opt.sm_skip ~= 'occlusion' then
         disp[2] = adcensus.interpolate_occlusion(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'occlusion')

      if sm_active and opt.sm_skip ~= 'occlusion' then
         disp[2] = adcensus.interpolate_mismatch(disp[2], outlier)
      end
      sm_active = sm_active and (opt.sm_terminate ~= 'mismatch')
   end
   if sm_active and opt.sm_skip ~= 'subpixel_enchancement' then
      disp[2] = adcensus.subpixel_enchancement(disp[2], vol, disp_max)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'subpixel_enchancement')

   if sm_active and opt.sm_skip ~= 'median' then
      disp[2] = adcensus.median2d(disp[2], 5)
   end
   sm_active = sm_active and (opt.sm_terminate ~= 'median')

   if sm_active and opt.sm_skip ~= 'bilateral' then
      disp[2] = adcensus.mean2d(disp[2], gaussian(opt.blur_sigma):cuda(), opt.blur_t)
   end

   return disp[2]
end
