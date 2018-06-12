require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')

-- This is important, need to keep it here otherwise it won't work
arch = 'fast'
dataset = 'aquabyte'
color_type = 'grayscale' -- also 'rgb'

require 'main_opt_aquabyte'

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

require 'main_utils_updated'
require 'main_predict_updated'

net_te = torch.load(opt.net_fname, 'ascii')[1]
net_te.modules[#net_te.modules] = nil
net_te2 = nn.StereoJoin(1):cuda()

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end


output_file_image = string.format("%s.png", opt.output_file_base) 

print(string.format("Generating depth map for frame pair: %s %s", opt.frame_path_left, opt.frame_path_right))

if not file_exists(opt.depth_map_out_bin_path_1) then
  predict(opt.frame_path_left, opt.frame_path_right, opt.depth_map_out_bin_path_1, opt.depth_map_out_bin_path_2)

  w = 1280
  h = 768

  if file_exists(opt.depth_map_out_bin_path_1) and not file_exists(opt.depth_map_out_png_path) then
    disp = torch.FloatTensor(torch.FloatStorage(opt.depth_map_out_bin_path_1)):view(1, 1, h, w)
    image.save(opt.depth_map_out_png_path, disp[1]:div(opt.disp_max))
  end
end

os.exit()

