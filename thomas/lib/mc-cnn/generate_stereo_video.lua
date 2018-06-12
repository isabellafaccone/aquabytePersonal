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

require 'lib/mc-cnn/main_opt_aquabyte'

opt = get_opt()

require 'cunn'
require 'cutorch'
require 'image'
require 'lib/mc-cnn/libadcensus'
require 'lib/mc-cnn/libcv'
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

luasql = require 'luasql.mysql'

require 'lib/mc-cnn/main_utils'
require 'lib/mc-cnn/main_predict'

net_te = torch.load(opt.net_fname, 'ascii')[1]
net_te.modules[#net_te.modules] = nil
net_te2 = nn.StereoJoin(1):cuda()

db_user = 'aquabyte'
db_password = 'bryton2017'
db_host = 'aquabyte-dev.co65lky0wxqd.us-west-1.rds.amazonaws.com'
db_port = 3306
db_name = 'aquabyte'
db_uri = string.format('mysql://%s:%s@%s:%i/%s', db_user, db_password, db_host, db_port, db_name)

env = assert (luasql.mysql())
con = assert (env:connect(db_name, db_user, db_password, db_host, db_port))

function split(inputstr, sep, type)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    if type == 'int' then
      t[i] = tonumber(str)
    else
      t[i] = str
    end 
    i = i + 1
  end
  return t
end

function rows (connection, sql_statement)
  local cursor = assert (connection:execute (sql_statement))
  return function ()
    return cursor:fetch()
  end
end

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end


frame_ids = split(opt.frame_ids, ',', 'int')
output_file_bases = split(opt.output_file_bases, ',', 'str')

for idx, frame_id in pairs(frame_ids) do
  local output_file_base = output_file_bases[idx]

  output_file_bin = string.format("%s.bin", output_file_base)
  output_file_image = string.format("%s.png", output_file_base) 

  -- retrieve a cursor
  cur = assert (con:execute(string.format('select * from frames where id=%i and is_bad=FALSE', frame_id)))

  -- print all rows
  row = cur:fetch ({}, "a") -- the rows will be indexed by field names

  count = 0

  while row do
    count = count + 1
    io.write(count)
    print (string.format ("Generating stereo for frame %i (%i)", row.frame_number, count))

    if not file_exists(output_file_bin) then
      local left = string.format('/%s', row.key_left)
      local right = string.format('/%s', row.key_right)


      predict(left, right, row.video_id, row.frame_number, frame_id, output_file_bin)

      w = tonumber(row.width)
      h = tonumber(row.height)

      if file_exists(output_file_bin) and not file_exists(output_file_image) then
        disp = torch.FloatTensor(torch.FloatStorage(output_file_bin)):view(1, 1, h, w)
        image.save(output_file_image, disp[1]:div(opt.disp_max))
      end
      
    end
    row = cur:fetch (row, "a")  -- reusing the table of results
  end
end

os.exit()

