#!/usr/bin/env python3
import json
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
from datetime import datetime,timedelta

AKPD_MIN = AKPD_OPTIMAL = AKPD_MAX = (1, 512, 512, 3)

class AkpdTRT(object):
    def __init__(self, model_path, config_path, device=0, fp16=False):
        self.is_active = False
        self._model_path = model_path
        self._config_path = config_path
        self._device_ix = device
        self._fp16 = fp16
        with config_path.open('r') as fp:
            self._config = json.load(fp)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(exc_type, exc_value, traceback)
        
    def open(self):
        if self.is_active:
            return

        cuda.init()

        self._device = cuda.Device(self._device_ix)
        self._context = self._device.make_context()

        self._trt_logger = trt.Logger(trt.Logger.VERBOSE)
        self._explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self._builder = trt.Builder(self._trt_logger).__enter__()
        self._network = self._builder.create_network(self._explicit_batch).__enter__()
        self._parser = trt.OnnxParser(self._network, self._trt_logger).__enter__() 

        self._builder.max_workspace_size = 1 << 30

        if self._fp16:
            assert self._builder.platform_has_fast_fp16, 'Cannot optimize for FP16 on this'
        self._builder.fp16_mode = self._fp16
    
        self._profile = self._builder.create_optimization_profile()
        self._profile.set_shape(self._config['input_name'], AKPD_MIN, AKPD_OPTIMAL, AKPD_MAX) 
        self._trt_config = self._builder.create_builder_config()
        self._trt_config.add_optimization_profile(self._profile)

        with open(self._model_path, 'rb') as model:
            if not self._parser.parse(model.read()):
                for error in range(self._parser.num_errors):
                    e = self._parser.get_error(error)
                    print(e)
                raise Exception('Could not load ONX model')
        shape = list(self._network.get_input(0).shape)
        shape[0] = -1
        self._network.get_input(0).shape = shape
        self._engine = self._builder.build_engine(self._network, self._trt_config)
        assert self._engine, 'Could not create TRT Engine'
        self._session = self._engine.create_execution_context()
        assert self._session, 'Could not create TRT execution context'

        for binding in self._engine:
            if self._engine.binding_is_input(binding):
                self._input_shape = self._engine.get_binding_shape(binding)
                self._input_size = trt.volume(self._input_shape) * self._engine.max_batch_size * np.dtype(np.float32).itemsize
                self._device_input = cuda.mem_alloc(self._input_size)
            else:
                self._output_shape = self._engine.get_binding_shape(binding)
                self._output_size = trt.volume(self._output_shape) * self._engine.max_batch_size
                self._host_output = cuda.pagelocked_empty(self._output_size, dtype=np.float32)
                self._device_output = cuda.mem_alloc(self._host_output.nbytes)

        self._stream = cuda.Stream()
        self.is_active = True

    def close(self, exc_type=None, exc_value=None, traceback=None):
        if not self.is_active:
            return

        if self._session:
            self._session.__exit__(exc_type,exc_value,traceback)
        if self._engine:
            self._engine.__exit__(exc_type,exc_value,traceback)
        if self._parser:
            self._parser.__exit__(exc_type,exc_value,traceback)
        if self._network:
            self._network.__exit__(exc_type,exc_value,traceback)
        if self._builder:
            self._builder.__exit__(exc_type,exc_value,traceback)
        if self._context:
            self._context.pop()
        
        self.is_active = False

    def _process(self, image):
        assert self.is_active, 'AKPD model not loaded'
        start_t = datetime.now()
        host_input = np.array(image, dtype=np.float32, order='C')
        cuda.memcpy_htod_async(self._device_input, host_input, self._stream)
        self._session.execute_async(bindings=[int(self._device_input), int(self._device_output)], stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(self._host_output, self._device_output , self._stream)
        self._stream.synchronize()
        output_data = torch.Tensor(self._host_output).reshape([int(i) for i in self._output_shape])
        final_stage_heatmap = output_data.squeeze()
        hm = final_stage_heatmap.cpu().detach().numpy().copy()
        tf_time = (datetime.now() - start_t).total_seconds()
        return hm, tf_time

    def process(self, left_image, right_image):
        left_hm, l_t = self._process(left_image)
        right_hm, r_t = self._process(right_image)
        return left_hm, right_hm, [l_t,r_t]
