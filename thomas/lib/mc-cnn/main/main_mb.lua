function load_mb()
	if opt.color == 'rgb' then
		n_input_plane = 3
	else
		n_input_plane = 1
	end
	err_at = 1

	if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'test_te' or opt.a == 'test_all' or opt.a == 'submit' then
		data_dir = ('data.mb.%s_%s'):format(opt.rect, opt.color)
		te = fromfile(('%s/te.bin'):format(data_dir))
		metadata = fromfile(('%s/meta.bin'):format(data_dir))
		nnz_tr = fromfile(('%s/nnz_tr.bin'):format(data_dir))
		nnz_te = fromfile(('%s/nnz_te.bin'):format(data_dir))
		fname_submit = {}
		for line in io.open(('%s/fname_submit.txt'):format(data_dir), 'r'):lines() do
			table.insert(fname_submit, line)
		end
		X = {}
		dispnoc = {}
		height = 1500
		width = 1000
		for n = 1,metadata:size(1) do
			local XX = {}
			light = 1
			while true do
				fname = ('%s/x_%d_%d.bin'):format(data_dir, n, light)
				if not paths.filep(fname) then
					break
				end
				table.insert(XX, fromfile(fname))
				light = light + 1
				if opt.a == 'test_te' or opt.a == 'submit' then
					break  -- we don't need to load training data
				end
			end
			table.insert(X, XX)

			fname = ('%s/dispnoc%d.bin'):format(data_dir, n)
			if paths.filep(fname) then
				table.insert(dispnoc, fromfile(fname))
			end
		end
	end
end