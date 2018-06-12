function load_kitti()
	height = 350
	width = 1242
	disp_max = 228
	n_te = dataset == 'kitti' and 195 or 200
	n_input_plane = 1
	err_at = 3

	if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'test_te' or opt.a == 'test_all' or opt.a == 'submit' then
		if opt.at == 1 then
			function load(fname)
				local X_12 = fromfile('data.kitti/' .. fname)
				local X_15 = fromfile('data.kitti2015/' .. fname)
				local X = torch.cat(X_12[{{1,194}}], X_15[{{1,200}}], 1)
				X = torch.cat(X, dataset == 'kitti' and X_12[{{195,389}}] or X_15[{{200,400}}], 1)
				return X
			end
			X0 = load('x0.bin')
			X1 = load('x1.bin')
			metadata = load('metadata.bin')

			dispnoc = torch.cat(fromfile('data.kitti/dispnoc.bin'), fromfile('data.kitti2015/dispnoc.bin'), 1)
			tr = torch.cat(fromfile('data.kitti/tr.bin'), fromfile('data.kitti2015/tr.bin'):add(194))
			te = dataset == 'kitti' and fromfile('data.kitti/te.bin') or fromfile('data.kitti2015/te.bin'):add(194)

			function load_nnz(fname)
				local X_12 = fromfile('data.kitti/' .. fname)
				local X_15 = fromfile('data.kitti2015/' .. fname)
				X_15[{{},1}]:add(194)
				return torch.cat(X_12, X_15, 1)
			end
			nnz_tr = load_nnz('nnz_tr.bin')
			nnz_te = load_nnz('nnz_te.bin')
			elseif dataset == 'kitti' then
				X0 = fromfile('data.kitti/x0.bin')
				X1 = fromfile('data.kitti/x1.bin')
				dispnoc = fromfile('data.kitti/dispnoc.bin')
				metadata = fromfile('data.kitti/metadata.bin')
				tr = fromfile('data.kitti/tr.bin')
				te = fromfile('data.kitti/te.bin')
				nnz_tr = fromfile('data.kitti/nnz_tr.bin')
				nnz_te = fromfile('data.kitti/nnz_te.bin')
				elseif dataset == 'kitti2015' then
					X0 = fromfile('data.kitti2015/x0.bin')
					X1 = fromfile('data.kitti2015/x1.bin')
					dispnoc = fromfile('data.kitti2015/dispnoc.bin')
					metadata = fromfile('data.kitti2015/metadata.bin')
					tr = fromfile('data.kitti2015/tr.bin')
					te = fromfile('data.kitti2015/te.bin')
					nnz_tr = fromfile('data.kitti2015/nnz_tr.bin')
					nnz_te = fromfile('data.kitti2015/nnz_te.bin')
				end
			end
		end