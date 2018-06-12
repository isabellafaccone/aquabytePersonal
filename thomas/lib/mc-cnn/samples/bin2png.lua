require 'cutorch'
require 'image'

video_id = 145

d = 400
--h = 992
--w = 1436
h = 576
w = 837

--[[print('Writing left.png')
left = torch.FloatTensor(torch.FloatStorage('/input/left.bin')):view(1, d, h, w):cuda()
>>>>>>> refactoring
_, left_ = left:min(2)
image.save('/output/left.png', left_[1]:float():div(d))

print('Writing right.png')
right = torch.FloatTensor(torch.FloatStorage('/output/right.bin')):view(1, d, h, w):cuda()
_, right_ = right:min(2)
image.save('/output/right.png', right_[1]:float():div(d))]]

--print('Writing disp.png')
for i = 248,248 do
	left = torch.FloatTensor(torch.FloatStorage(('/videos/%d/stereo/%d-left.bin'):format(video_id, i))):view(1, d, h, w):cuda()
	_, left_ = left:min(2)
	image.save(('/videos/%d/stereo/%d-left.png'):format(video_id, i), left_[1]:float():div(d))

	right = torch.FloatTensor(torch.FloatStorage(('/videos/%d/stereo/%d-right.bin'):format(video_id, i))):view(1, d, h, w):cuda()
	_, right_ = right:min(2)
	image.save(('/videos/%d/stereo/%d-right.png'):format(video_id, i), right_[1]:float():div(d))

	disp = torch.FloatTensor(torch.FloatStorage(('/videos/%d/stereo/%d.bin'):format(video_id, i))):view(1, 1, h, w)
	image.save(('/videos/%d/stereo/%d.png'):format(video_id, i), disp[1]:div(d))
end
