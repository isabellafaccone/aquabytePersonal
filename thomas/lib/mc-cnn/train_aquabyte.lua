#! /usr/bin/env luajit

require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')
dataset = table.remove(arg, 1)
arch = table.remove(arg, 1)
assert(dataset == 'aquabyte')
assert(arch == 'fast' or arch == 'slow' or arch == 'ad' or arch == 'census')

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

cmd_str = dataset .. '_' .. arch
for i = 1,#arg do
   cmd_str = cmd_str .. '_' .. arg[i]
end

require 'main_utils'

-- load training data
height = 480
width = 640
disp_max = 228
n_te = 120 --910
n_input_plane = 1
err_at = 3

if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'test_te' or opt.a == 'test_all' or opt.a == 'submit' then
   if opt.at == 1 then
      function load(fname)
         local X_15 = fromfile('/videos/32/' .. fname)
         local X = torch.cat(X_15[{{1,n_te-1}}], 1)
         X = torch.cat(X, X_15[{{n_te-1,2*n_te}}], 1)
         return X
      end
      X0 = load('x0.bin')
      X1 = load('x1.bin')
      metadata = load('metadata.bin')

      dispnoc = torch.cat(fromfile('/videos/32/dispnoc.bin'), 1)
      tr = fromfile('/videos/32/tr.bin')
      te = fromfile('/videos/32/te.bin')

      function load_nnz(fname)
         local X_15 = fromfile('/videos/32/' .. fname)
         return torch.cat(X_15, 1)
      end

      nnz_tr = load_nnz('nnz_tr.bin')
      nnz_te = load_nnz('nnz_te.bin')
   elseif dataset == 'aquabyte' then
      X0 = fromfile('/videos/32/x0.bin')
      X1 = fromfile('/videos/32/x1.bin')
      dispnoc = fromfile('/videos/32/dispnoc.bin')
      metadata = fromfile('/videos/32/metadata.bin')
      tr = fromfile('/videos/32/tr.bin')
      te = fromfile('/videos/32/te.bin')
      nnz_tr = fromfile('/videos/32/nnz_tr.bin')
      nnz_te = fromfile('/videos/32/nnz_te.bin')
   end
end

if opt.a == 'train_tr' or opt.a == 'train_all' or opt.a == 'time' then
   function mul32(a,b)
      return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
   end

   function make_patch(src, dst, dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
      local m = {1, 0, -dim4, 0, 1, -dim3}
      m = mul32({1, 0, trans[1], 0, 1, trans[2]}, m) -- translate
      m = mul32({scale[1], 0, 0, 0, scale[2], 0}, m) -- scale
      local c = math.cos(phi)
      local s = math.sin(phi)
      m = mul32({c, s, 0, -s, c, 0}, m) -- rotate
      m = mul32({1, hshear, 0, 0, 1, 0}, m) -- shear
      m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
      m = torch.FloatTensor(m)
      cv.warp_affine(src, dst, m)
      dst:mul(contrast):add(brightness)
   end

   -- subset training dataset
   if opt.subset < 1 then
      function sample(xs, p)
         local perm = torch.randperm(xs:nElement()):long()
         return xs:index(1, perm[{{1, xs:size(1) * p}}])
      end

      local tr_subset = sample(tr, opt.subset)

      local nnz_tr_output = torch.FloatTensor(nnz_tr:size()):zero()
      local t = adcensus.subset_dataset(tr_subset, nnz_tr, nnz_tr_output);
      nnz_tr = nnz_tr_output[{{1,t}}]
   end
   collectgarbage()

   if opt.a == 'train_all' then
      nnz = torch.cat(nnz_tr, nnz_te, 1)
   elseif opt.a == 'train_tr' or opt.a == 'time' then
      nnz = nnz_tr
   end

   if opt.a ~= 'time' then
      perm = torch.randperm(nnz:size(1))
   end

   local fm = torch.totable(torch.linspace(opt.fm, opt.fm, opt.l1):int())

   -- network for training
   if arch == 'slow' then
      net_tr = nn.Sequential()
      for i = 1,#fm do
         net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
         net_tr:add(cudnn.ReLU(true))
      end
      net_tr:add(nn.Reshape(opt.bs, 2 * fm[#fm]))
      for i = 1,opt.l2 do
         net_tr:add(nn.Linear(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
         net_tr:add(cudnn.ReLU(true))
      end
      net_tr:add(nn.Linear(opt.nh2, 1))
      net_tr:add(cudnn.Sigmoid(false))
      net_tr:cuda()
      criterion = nn.BCECriterion2():cuda()

      -- network for testing (make sure it's synched with net_tr)
      local pad = (opt.ks - 1) / 2
      net_te = nn.Sequential()
      for i = 1,#fm do
         net_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
         net_te:add(cudnn.ReLU(true))
      end
      net_te:cuda()

      net_te2 = nn.Sequential()
      for i = 1,opt.l2 do
         net_te2:add(nn.SpatialConvolution1_fw(i == 1 and 2 * fm[#fm] or opt.nh2, opt.nh2))
         net_te2:add(cudnn.ReLU(true))
      end
      net_te2:add(nn.SpatialConvolution1_fw(opt.nh2, 1))
      net_te2:add(cudnn.Sigmoid(true))
      net_te2:cuda()

      -- tie weights
      net_te_all = {}
      for i, v in ipairs(net_te.modules) do table.insert(net_te_all, v) end
      for i, v in ipairs(net_te2.modules) do table.insert(net_te_all, v) end

      local finput = torch.CudaTensor()
      local i_tr = 1
      local i_te = 1
      while i_tr <= net_tr:size() do
         local module_tr = net_tr:get(i_tr)
         local module_te = net_te_all[i_te]

         local skip = {['nn.Reshape']=1, ['nn.Dropout']=1}
         while skip[torch.typename(module_tr)] do
            i_tr = i_tr + 1
            module_tr = net_tr:get(i_tr)
         end

         if module_tr.weight then
            -- print(('tie weights of %s and %s'):format(torch.typename(module_te), torch.typename(module_tr)))
            assert(module_te.weight:nElement() == module_tr.weight:nElement())
            assert(module_te.bias:nElement() == module_tr.bias:nElement())
            module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
            module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
         end

         i_tr = i_tr + 1
         i_te = i_te + 1
      end
   elseif arch == 'fast' then
      net_tr = nn.Sequential()
      for i = 1,#fm do
         net_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
         if i < #fm then
            net_tr:add(cudnn.ReLU(true))
         end
      end
      net_tr:add(nn.Normalize2())
      net_tr:add(nn.StereoJoin1())
      net_tr:cuda()

      net_te = net_tr:clone('weight', 'bias')
      net_te.modules[#net_te.modules] = nn.StereoJoin(1):cuda()
      for i = 1,#net_te.modules do
         local m = net_te:get(i)
         if torch.typename(m) == 'cudnn.SpatialConvolution' then
            m.padW = 1
            m.padH = 1
         end
      end

      criterion = nn.Margin2(opt.m, opt.pow):cuda()
   end

   print_net(net_tr)

   params = {}
   grads = {}
   momentums = {}
   for i = 1,net_tr:size() do
      local m = net_tr:get(i)
      if m.weight then
         m.weight_v = torch.CudaTensor(m.weight:size()):zero()
         table.insert(params, m.weight)
         table.insert(grads, m.gradWeight)
         table.insert(momentums, m.weight_v)
      end
      if m.bias then
         m.bias_v = torch.CudaTensor(m.bias:size()):zero()
         table.insert(params, m.bias)
         table.insert(grads, m.gradBias)
         table.insert(momentums, m.bias_v)
      end
   end

   ws = get_window_size(net_tr)
   x_batch_tr = torch.CudaTensor(opt.bs * 2, n_input_plane, ws, ws)
   y_batch_tr = torch.CudaTensor(opt.bs)
   x_batch_tr_ = torch.FloatTensor(x_batch_tr:size())
   y_batch_tr_ = torch.FloatTensor(y_batch_tr:size())

   time = sys.clock()
   for epoch = 1,14 do
      if opt.a == 'time' then
         break
      end
      if epoch == 12 then
         opt.lr = opt.lr / 10
      end

      local err_tr = 0
      local err_tr_cnt = 0
      for t = 1,nnz:size(1) - opt.bs/2,opt.bs/2 do
         for i = 1,opt.bs/2 do
            d_pos = torch.uniform(-opt.true1, opt.true1)
            d_neg = torch.uniform(opt.false1, opt.false2)
            if torch.uniform() < 0.5 then
               d_neg = -d_neg
            end

            assert(opt.hscale <= 1 and opt.scale <= 1)
            local s = torch.uniform(opt.scale, 1)
            local scale = {s * torch.uniform(opt.hscale, 1), s}
            if opt.hflip == 1 and torch.uniform() < 0.5 then
               scale[1] = -scale[1]
            end
            if opt.vflip == 1 and torch.uniform() < 0.5 then
               scale[2] = -scale[2]
            end
            local hshear = torch.uniform(-opt.hshear, opt.hshear)
            local trans = {torch.uniform(-opt.trans, opt.trans), torch.uniform(-opt.trans, opt.trans)}
            local phi = torch.uniform(-opt.rotate * math.pi / 180, opt.rotate * math.pi / 180)
            local brightness = torch.uniform(-opt.brightness, opt.brightness)
            assert(opt.contrast >= 1 and opt.d_contrast >= 1)
            local contrast = torch.uniform(1 / opt.contrast, opt.contrast)

            local scale_ = {scale[1] * torch.uniform(opt.d_hscale, 1), scale[2]}
            local hshear_ = hshear + torch.uniform(-opt.d_hshear, opt.d_hshear)
            local trans_ = {trans[1], trans[2] + torch.uniform(-opt.d_vtrans, opt.d_vtrans)}
            local phi_ = phi + torch.uniform(-opt.d_rotate * math.pi / 180, opt.d_rotate * math.pi / 180)
            local brightness_ = brightness + torch.uniform(-opt.d_brightness, opt.d_brightness)
            local contrast_ = contrast * torch.uniform(1 / opt.d_contrast, opt.d_contrast)

            local ind = perm[t + i - 1]
            img = nnz[{ind, 1}]
            dim3 = nnz[{ind, 2}]
            dim4 = nnz[{ind, 3}]
            d = nnz[{ind, 4}]
            
            x0 = X0[img]
            x1 = X1[img]

            make_patch(x0, x_batch_tr_[i * 4 - 3], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
            make_patch(x1, x_batch_tr_[i * 4 - 2], dim3, dim4 - d + d_pos, scale_, phi_, trans_, hshear_, brightness_, contrast_)
            make_patch(x0, x_batch_tr_[i * 4 - 1], dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
            make_patch(x1, x_batch_tr_[i * 4 - 0], dim3, dim4 - d + d_neg, scale_, phi_, trans_, hshear_, brightness_, contrast_)

            y_batch_tr_[i * 2 - 1] = 0
            y_batch_tr_[i * 2] = 1
         end

         x_batch_tr:copy(x_batch_tr_)
         y_batch_tr:copy(y_batch_tr_)

         for i = 1,#params do
            grads[i]:zero()
         end

         net_tr:forward(x_batch_tr)
         local err = criterion:forward(net_tr.output, y_batch_tr)
         if err >= 0 and err < 100 then
            err_tr = err_tr + err
            err_tr_cnt = err_tr_cnt + 1
         else
            print(('WARNING! err=%f'):format(err))
         end

         criterion:backward(net_tr.output, y_batch_tr)
         net_tr:backward(x_batch_tr, criterion.gradInput)

         for i = 1,#params do
            momentums[i]:mul(opt.mom):add(-opt.lr, grads[i])
            params[i]:add(momentums[i])
         end
      end

      if opt.debug then
         save_net(epoch)
      end
      print(epoch, err_tr / err_tr_cnt, opt.lr, sys.clock() - time)
      collectgarbage()
   end
   opt.net_fname = save_net(0)
   if opt.a == 'train_tr' then
      opt.a = 'test_te'
   elseif opt.a == 'train_all' then
      opt.a = 'submit'
   end
   collectgarbage()
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

if opt.a == 'submit' then
   os.execute('rm -rf out/*')
   if dataset == 'aquabyte' then
      os.execute('mkdir out/disp_0')
   end
   
   examples = torch.totable(torch.range(X0:size(1) - n_te + 1, X0:size(1)))
elseif opt.a == 'test_te' then
   examples = torch.totable(te)
elseif opt.a == 'test_all' then
   examples = torch.totable(torch.cat(tr, te))
end

if opt.a == 'time' then
   -- TODO: this might not work... because it was from kitti not kitti2015   
   x_batch = torch.CudaTensor(2, 1, 350, 1242)
   disp_max = 228

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
   img_height = metadata[{i,1}]
   img_width = metadata[{i,2}]
   id = metadata[{i,3}]
   x0 = X0[{{i},{},{},{1,img_width}}]
   x1 = X1[{{i},{},{},{1,img_width}}]

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
      pred_img = torch.FloatTensor(img_height, img_width):zero()
      pred_img:narrow(1, img_height - height + 1, height):copy(pred[{1,1}])
     
      path = 'out/disp_0'
      
      adcensus.writePNG16(pred_img, img_height, img_width, ("%s/%06d_10.png"):format(path, id))
   else
      assert(not isnan(pred:sum()))
      
      actual = dispnoc[{i,{},{},{1,img_width}}]:cuda()
      
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

         gt = dispnoc[{{i},{},{},{1,img_width}}]:cuda()
         
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