function predict(left, right, video_id, frame_number, frame_id, output_file)

	print (('Predicting %s, left: %s, right: %s, frame_id: %i'):format(output_file, left, right, frame_id))
	
	x0 = image.load(left, nil, 'byte'):float()
	x1 = image.load(right, nil, 'byte'):float()

	if x0:size(1) == 3 then
		assert(x1:size(1) == 3)
		x0 = image.rgb2y(x0)
		x1 = image.rgb2y(x1)
	end
	disp_max = opt.disp_max

	x0:add(-x0:mean()):div(x0:std())
	x1:add(-x1:mean()):div(x1:std())

	x_batch = torch.CudaTensor(2, 1, x0:size(2), x0:size(3))
	x_batch[1]:copy(x0)
	x_batch[2]:copy(x1)
	disp = stereo_predict(video_id, frame_number, x_batch, 0)

	print(('Writing %s'):format(output_file))
	
	torch.DiskFile(output_file, 'w'):binary():writeFloat(disp:float():storage())
end