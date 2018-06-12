function get_opt()
	cmd = torch.CmdLine()
	cmd:option('-gpu', 1, 'gpu id')
	cmd:option('-seed', 42, 'random seed')
	cmd:option('-debug', false)
	cmd:option('-d', 'aquabyte')
	cmd:option('-a', 'train_tr | train_all | test_te | test_all | submit | time | predict', 'train')
	cmd:option('-net_fname', '')
	cmd:option('-make_cache', false)
	cmd:option('-use_cache', false)
	cmd:option('-print_args', false)
	cmd:option('-sm_terminate', '', 'terminate the stereo method after this step')
	cmd:option('-sm_skip', '', 'which part of the stereo method to skip')
	cmd:option('-tiny', false)
	cmd:option('-subset', 1)

	cmd:option('-left', '')
	cmd:option('-right', '')
	cmd:option('-disp_max', '')
	cmd:option('-video_id', 0)

	cmd:option('-hflip', 0)
	cmd:option('-vflip', 0)
	cmd:option('-rotate', 7) -- different from other dataset
	cmd:option('-hscale', 0.9) -- different from other dataset
	cmd:option('-scale', 1) -- different from other dataset
	cmd:option('-trans', 0)
	cmd:option('-hshear', 0.1)
	cmd:option('-brightness', 0.7) -- different from other dataset
	cmd:option('-contrast', 1.3) -- different from other dataset
	cmd:option('-d_vtrans', 0) -- different from other dataset
	cmd:option('-d_rotate', 0) -- different from other dataset
	cmd:option('-d_hscale', 1) -- different from other dataset
	cmd:option('-d_hshear', 0) -- different from other dataset
	cmd:option('-d_brightness', 0.3) -- different from other dataset
	cmd:option('-d_contrast', 1) -- different from other dataset

	cmd:option('-rect', 'imperfect')
	cmd:option('-color', 'gray')

	if arch == 'slow' then
		cmd:option('-at', 0) -- omitted from other dataset

		cmd:option('-l1', 4) -- different from other dataset
		cmd:option('-fm', 112)
		cmd:option('-ks', 3)
		cmd:option('-l2', 4) -- different from other dataset
		cmd:option('-nh2', 384)
		cmd:option('-lr', 0.003)
		cmd:option('-bs', 128)
		cmd:option('-mom', 0.9)
		cmd:option('-true1', 1) -- different from other dataset
		cmd:option('-false1', 4) -- different from other dataset
		cmd:option('-false2', 10) -- different from other dataset

		-- these were in the mb dataset
		--cmd:option('-ds', 2001)
		--cmd:option('-d_exp', 0.2)
		--cmd:option('-d_light', 0.2)

		if color_type == 'grayscale' then
			cmd:option('-L1', 5)
			cmd:option('-cbca_i1', 2)
			cmd:option('-cbca_i2', 0)
			cmd:option('-tau1', 0.13)
			cmd:option('-pi1', 1.32)
			cmd:option('-pi2', 24.25)
			cmd:option('-sgm_i', 1)
			cmd:option('-sgm_q1', 3)
			cmd:option('-sgm_q2', 2)
			cmd:option('-alpha1', 2)
			cmd:option('-tau_so', 0.08)
			cmd:option('-blur_sigma', 5.99)
			cmd:option('-blur_t', 6)
		else
			cmd:option('-L1', 5) -- different from other dataset
			cmd:option('-cbca_i1', 2)
			cmd:option('-cbca_i2', 4) -- different from other dataset
			cmd:option('-tau1', 0.03) -- different from other dataset
			cmd:option('-pi1', 2.3) -- different from other dataset
			cmd:option('-pi2', 24.25) -- different from other dataset
			cmd:option('-sgm_i', 1)
			cmd:option('-sgm_q1', 3) -- different from other dataset
			cmd:option('-sgm_q2', 2)
			cmd:option('-alpha1', 1.75) -- different from other dataset
			cmd:option('-tau_so', 0.08) -- different from other dataset
			cmd:option('-blur_sigma', 5.99) -- different from other dataset
			cmd:option('-blur_t', 5) -- different from other dataset
		end
	elseif arch == 'census' then
		cmd:option('-L1', 0) -- different from other dataset
		cmd:option('-cbca_i1', 4) -- different from other dataset
		cmd:option('-cbca_i2', 8)
		cmd:option('-tau1', 0.01) -- different from other dataset
		cmd:option('-pi1', 4)
		cmd:option('-pi2', 128.00) -- different from other dataset
		cmd:option('-sgm_i', 1)
		cmd:option('-sgm_q1', 3) -- different from other dataset
		cmd:option('-sgm_q2', 3.5) -- different from other dataset
		cmd:option('-alpha1', 1.25) -- different from other dataset
		cmd:option('-tau_so', 1.0) -- different from other dataset
		cmd:option('-blur_sigma', 7.74) -- different from other dataset
		cmd:option('-blur_t', 6) -- different from other dataset
	elseif arch == 'ad' then
		cmd:option('-L1', 3) -- different from other dataset
		cmd:option('-cbca_i1', 0)
		cmd:option('-cbca_i2', 4)
		cmd:option('-tau1', 0.03) -- different from other dataset
		cmd:option('-pi1', 0.76) -- different from other dataset
		cmd:option('-pi2', 13.93) -- different from other dataset
		cmd:option('-sgm_i', 1)
		cmd:option('-sgm_q1', 3.5) -- different from other dataset
		cmd:option('-sgm_q2', 2) -- different from other dataset
		cmd:option('-alpha1', 2.5)
		cmd:option('-tau_so', 0.01) -- different from other dataset
		cmd:option('-blur_sigma', 7.74)
		cmd:option('-blur_t', 6) -- different from other dataset
	elseif arch == 'fast' then
		if color_type == 'grayscale' then
			cmd:option('-at', 0)
			cmd:option('-m', 0.2, 'margin')
			cmd:option('-pow', 1)

			cmd:option('-l1', 4)
			cmd:option('-fm', 64)
			cmd:option('-ks', 3)
			cmd:option('-lr', 0.002)
			cmd:option('-bs', 128)
			cmd:option('-mom', 0.9)
			cmd:option('-true1', 1)
			cmd:option('-false1', 4)
			cmd:option('-false2', 10)

			cmd:option('-L1', 0)
			cmd:option('-cbca_i1', 0)
			cmd:option('-cbca_i2', 0)
			cmd:option('-tau1', 0)
			cmd:option('-pi1', 10)
			cmd:option('-pi2', 70)
			cmd:option('-sgm_i', 1)
			cmd:option('-sgm_q1', 5)
			cmd:option('-sgm_q2', 2.5)
			cmd:option('-alpha1', 1.5)
			cmd:option('-tau_so', 0.02)
			cmd:option('-blur_sigma', 7.74)
			cmd:option('-blur_t', 5)
		else
			cmd:option('-at', 0) -- omitted from the other dataset
			cmd:option('-m', 0.2, 'margin')
			cmd:option('-pow', 1)

			cmd:option('-l1', 4) -- different from other dataset
			cmd:option('-fm', 64)
			cmd:option('-ks', 3)
			cmd:option('-lr', 0.002)
			cmd:option('-bs', 128)
			cmd:option('-mom', 0.9)
			cmd:option('-true1', 1) -- different from other dataset
			cmd:option('-false1', 4) -- different from other dataset
			cmd:option('-false2', 10) -- different from other dataset

			cmd:option('-L1', 0)
			cmd:option('-cbca_i1', 0)
			cmd:option('-cbca_i2', 0)
			cmd:option('-tau1', 0)
			cmd:option('-pi1', 2.3) -- vary this
			cmd:option('-pi2', 18.38) -- vary this
			cmd:option('-sgm_i', 1)
			cmd:option('-sgm_q1', 3) -- different from other dataset
			cmd:option('-sgm_q2', 2) -- vary this
			cmd:option('-alpha1', 1.25) -- vary this
			cmd:option('-tau_so', 0.08) -- vary this
			cmd:option('-blur_sigma', 4.64) -- vary this
			cmd:option('-blur_t', 5) -- different from other dataset
		end
		-- these were in the mb dataset
		--cmd:option('-ds', 2001)
		--cmd:option('-d_exp', 0.2)
		--cmd:option('-d_light', 0.2)
	end

	opt = cmd:parse(arg)

	if opt.print_args then
		print((opt.ks - 1) * opt.l1 + 1, 'arch_patch_size')
		print(opt.l1, 'arch1_num_layers')
		print(opt.fm, 'arch1_num_feature_maps')
		print(opt.ks, 'arch1_kernel_size')
		print(opt.l2, 'arch2_num_layers')
		print(opt.nh2, 'arch2_num_units_2')
		print(opt.false1, 'dataset_neg_low')
		print(opt.false2, 'dataset_neg_high')
		print(opt.true1, 'dataset_pos_low')
		print(opt.tau1, 'cbca_intensity')
		print(opt.L1, 'cbca_distance')
		print(opt.cbca_i1, 'cbca_num_iterations_1')
		print(opt.cbca_i2, 'cbca_num_iterations_2')
		print(opt.pi1, 'sgm_P1')
		print(opt.pi1 * opt.pi2, 'sgm_P2')
		print(opt.sgm_q1, 'sgm_Q1')
		print(opt.sgm_q1 * opt.sgm_q2, 'sgm_Q2')
		print(opt.alpha1, 'sgm_V')
		print(opt.tau_so, 'sgm_intensity')
		print(opt.blur_sigma, 'blur_sigma')
		print(opt.blur_t, 'blur_threshold')
		os.exit()
	end

	return opt
end