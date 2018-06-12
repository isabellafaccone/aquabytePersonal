require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')

require 'main_opt'

opt = get_opt()

require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'
cudnn.benchmark = true

include('Margin2.lua')
include('Normalize2.lua')
include('BCECriterion2.lua')
include('StereoJoin.lua')
include('StereoJoin1.lua')
include('SpatialConvolution1_fw.lua')
-- include('SpatialLogSoftMax.lua')

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

require 'main_utils'
require 'main_predict'

net_te = torch.load(opt.net_fname, 'ascii')[1]
net_te.modules[#net_te.modules] = nil
net_te2 = nn.StereoJoin(1):cuda()

predict(opt.left, opt.right)