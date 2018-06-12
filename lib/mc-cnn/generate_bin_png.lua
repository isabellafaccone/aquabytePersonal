require 'cutorch'
require 'image'

cmd = torch.CmdLine()
cmd:option('-disp_max', '')
cmd:option('-video_id', 0)
opt = cmd:parse(arg)

h = 576
w = 837

luasql = require 'luasql.mysql'

db_user = 'aquabyte'
db_password = 'bryton2017'
db_host = 'aquabyte.co65lky0wxqd.us-west-1.rds.amazonaws.com'
db_port = 3306
db_name = 'aquabyte'
db_uri = string.format('mysql://%s:%s@%s:%i/%s', db_user, db_password, db_host, db_port, db_name)

env = assert (luasql.mysql())
con = assert (env:connect(db_name, db_user, db_password, db_host, db_port))

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

-- retrieve a cursor
cur = assert (con:execute(string.format('select frame_number from frames where video_id=%i and skip=FALSE', opt.video_id)))
-- print all rows
row = cur:fetch ({}, "a") -- the rows will be indexed by field names

count = 0

while row do
  count = count + 1

  print (string.format ("Generating stereo for frame %i (%i)", row.frame_number, count))

	--[[left = torch.FloatTensor(torch.FloatStorage(('/videos/%d/stereo/left_%d.bin'):format(video_id, frame_number))):view(1, d, h, w):cuda()
	_, left_ = left:min(2)
	image.save(('/videos/%d/stereo/%d-left.png'):format(video_id, frame_number), left_[1]:float():div(d))

	right = torch.FloatTensor(torch.FloatStorage(('/videos/%d/stereo/right_%d.bin'):format(video_id, frame_number))):view(1, d, h, w):cuda()
	_, right_ = right:min(2)
	image.save(('/videos/%d/stereo/%d-right.png'):format(video_id, frame_number), right_[1]:float():div(d))]]--

	local infile = ('/videos/%i/stereo/%d.bin'):format(opt.video_id, row.frame_number)
	local outfile = ('/videos/%i/stereo/%d.png'):format(opt.video_id, row.frame_number)

  if file_exists(infile) and not file_exists(outfile) then
		disp = torch.FloatTensor(torch.FloatStorage(infile)):view(1, 1, h, w)
		image.save(('/videos/%d/stereo/%d.png'):format(opt.video_id, row.frame_number), disp[1]:div(opt.disp_max))
	end

	row = cur:fetch (row, "a")
end

os.exit()