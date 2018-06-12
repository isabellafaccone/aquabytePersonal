require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')
dataset = table.remove(arg, 1)
arch = table.remove(arg, 1)
assert(dataset == 'kitti' or dataset == 'kitti2015' or dataset == 'mb')
assert(arch == 'fast' or arch == 'slow' or arch == 'ad' or arch == 'census')

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

cmd_str = dataset .. '_' .. arch
for i = 1,#arg do
   cmd_str = cmd_str .. '_' .. arg[i]
end

require 'main_utils'
require 'main_kitti'
require 'main_mb'
require 'main_train'
require 'main_predict'

-- load training data
if dataset == 'kitti' or dataset == 'kitti2015' then
   load_kitti()
elseif dataset == 'mb' then
   load_mb()
end

if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'time' then
   train()
end

if not opt.use_cache then
   if arch == 'slow' then
      net = torch.load(opt.net_fname, 'ascii')
      net_te = net[1]
      net_te2 = net[2]
   elseif arch == 'fast' then
      net_te = torch.load(opt.net_fname, 'ascii')[1]
      net_te.modules[#net_te.modules] = nil
      net_te2 = nn.StereoJoin(1):cuda()
   end
end

x_batch_te1 = torch.CudaTensor()
x_batch_te2 = torch.CudaTensor()

if opt.a == 'predict' then
   predict()
end

if opt.a == 'submit' then
   os.execute('rm -rf out/*')
   if dataset == 'kitti2015' then
      os.execute('mkdir out/disp_0')
   end
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(torch.range(X0:size(1) - n_te + 1, X0:size(1)))
   elseif dataset == 'mb' then
      examples = {}
      -- for i = #X - 14, #X do
      for i = #X - 29, #X do
         table.insert(examples, {i, 2})
      end
   end
elseif opt.a == 'test_te' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(te)
   elseif dataset == 'mb' then
      examples = {}
      for i = 1,te:nElement() do
         table.insert(examples, {te[i], 2})
      end
      table.insert(examples, {5, 3})
      table.insert(examples, {5, 4})
   end
elseif opt.a == 'test_all' then
   if dataset == 'kitti' or dataset == 'kitti2015' then
      examples = torch.totable(torch.cat(tr, te))
   elseif dataset == 'mb' then
      assert(false, 'test_all not supported on Middlebury.')
   end
end

if opt.a == 'time' then
   if opt.tiny then
      x_batch = torch.CudaTensor(2, 1, 240, 320)
      disp_max = 32
   elseif dataset == 'kitti' then
      x_batch = torch.CudaTensor(2, 1, 350, 1242)
      disp_max = 228
   elseif dataset == 'mb' then
      x_batch = torch.CudaTensor(2, 1, 1000, 1500)
      disp_max = 200
   end

   N = arch == 'fast' and 30 or 3
   runtime_min = 1 / 0
   for _ = 1,N do
      cutorch.synchronize()
      sys.tic()

      stereo_predict(x_batch, id)

      cutorch.synchronize()
      runtime = sys.toc()
      if runtime < runtime_min then
         runtime_min = runtime
      end
      collectgarbage()
   end
   print(runtime_min)

   os.exit()
end

err_sum = 0
x_batch = torch.CudaTensor(2, 1, height, width)
pred_good = torch.CudaTensor()
pred_bad = torch.CudaTensor()
for _, i in ipairs(examples) do
   if dataset == 'kitti' or dataset == 'kitti2015' then
      img_height = metadata[{i,1}]
      img_width = metadata[{i,2}]
      id = metadata[{i,3}]
      x0 = X0[{{i},{},{},{1,img_width}}]
      x1 = X1[{{i},{},{},{1,img_width}}]
   elseif dataset == 'mb' then
      i, right = table.unpack(i)
      id = ('%d_%d'):format(i, right)
      disp_max = metadata[{i,3}]
      x0 = X[i][1][{{1}}]
      x1 = X[i][1][{{right}}]
   end

   x_batch:resize(2, 1, x0:size(3), x0:size(4))
   x_batch[1]:copy(x0)
   x_batch[2]:copy(x1)

   collectgarbage()
   cutorch.synchronize()
   sys.tic()
   pred = stereo_predict(x_batch, id)
   cutorch.synchronize()
   runtime = sys.toc()

   if opt.a == 'submit' then
      if dataset == 'kitti' or dataset == 'kitti2015' then
         pred_img = torch.FloatTensor(img_height, img_width):zero()
         pred_img:narrow(1, img_height - height + 1, height):copy(pred[{1,1}])
        
         if dataset == 'kitti' then
            path = 'out'
         elseif dataset == 'kitti2015' then
            path = 'out/disp_0'
         end
         adcensus.writePNG16(pred_img, img_height, img_width, ("%s/%06d_10.png"):format(path, id))
      elseif dataset == 'mb' then
         -- savePNG(('tmp/fos_%d.png'):format(i), pred)
         base = 'out/' .. fname_submit[i - (#X - #fname_submit)]
         os.execute('mkdir -p ' .. base)
         local method_name = 'MC-CNN-' .. (arch == 'fast' and 'fst' or 'acrt' )
         adcensus.writePFM(image.vflip(pred[{1,1}]:float()), base .. '/disp0' .. method_name .. '.pfm')
         local f = io.open(base .. '/time' .. method_name .. '.txt', 'w')
         f:write(tostring(runtime))
         f:close()
      end
   else
      assert(not isnan(pred:sum()))
      if dataset == 'kitti' or dataset == 'kitti2015' then
         actual = dispnoc[{i,{},{},{1,img_width}}]:cuda()
      elseif dataset == 'mb' then
         actual = dispnoc[i]:cuda()
      end
      pred_good:resizeAs(actual)
      pred_bad:resizeAs(actual)
      mask = torch.CudaTensor():resizeAs(actual):ne(actual, 0)
      actual:add(-1, pred):abs()
      pred_bad:gt(actual, err_at):cmul(mask)
      pred_good:le(actual, err_at):cmul(mask)
      local err = pred_bad:sum() / mask:sum()
      err_sum = err_sum + err
      print(runtime, err)

      if opt.debug then
         local img_pred = torch.Tensor(1, 3, pred:size(3), pred:size(4))
         adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
         if x0:size(2) == 1 then
            x0 = torch.repeatTensor(x0:cuda(), 1, 3, 1, 1)
         end
         img_err = x0:mul(50):add(150):div(255)
         img_err[{1,1}]:add( 0.5, pred_bad)
         img_err[{1,2}]:add(-0.5, pred_bad)
         img_err[{1,3}]:add(-0.5, pred_bad)
         img_err[{1,1}]:add(-0.5, pred_good)
         img_err[{1,2}]:add( 0.5, pred_good)
         img_err[{1,3}]:add(-0.5, pred_good)

         if dataset == 'kitti' or dataset == 'kitti2015' then
            gt = dispnoc[{{i},{},{},{1,img_width}}]:cuda()
         elseif dataset == 'mb' then
            gt = dispnoc[i]:cuda():resize(1, 1, pred:size(3), pred:size(4))
         end
         local img_gt = torch.Tensor(1, 3, pred:size(3), pred:size(4)):zero()
         adcensus.grey2jet(gt:double():add(1)[{1,1}]:div(disp_max):double(), img_gt)
         img_gt[{1,3}]:cmul(mask:double())

         image.save(('tmp/%s_%s_gt.png'):format(dataset, id), img_gt[1])
         image.save(('tmp/%s_%s_%s_pred.png'):format(dataset, arch, id), img_pred[1])
         image.save(('tmp/%s_%s_%s_err.png'):format(dataset, arch, id), img_err[1])

--         adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
--         if x0:size(2) == 1 then
--            x0 = torch.repeatTensor(x0:cuda(), 1, 3, 1, 1)
--         end
--         img_err = x0:mul(50):add(150):div(255)
--         img_err[{1,1}]:add( 0.3, pred_bad)
--         img_err[{1,2}]:add(-0.3, pred_bad)
--         img_err[{1,3}]:add(-0.3, pred_bad)
--         img_err[{1,1}]:add(-0.3, pred_good)
--         img_err[{1,2}]:add( 0.3, pred_good)
--         img_err[{1,3}]:add(-0.3, pred_good)
--
--         img = torch.Tensor(3, pred:size(3) * 2, pred:size(4))
--         img:narrow(2, 0 * pred:size(3) + 1, pred:size(3)):copy(img_pred)
--         img:narrow(2, 1 * pred:size(3) + 1, pred:size(3)):copy(img_err)
--
--         image.savePNG(('tmp/err_%d_%.5f_%s.png'):format(opt.gpu, err, id), img)
      end
   end
end

if opt.a == 'submit' then
   -- zip
   os.execute('cd out; zip -r submission.zip . -x .empty')
else
   print(err_sum / #examples)
end
