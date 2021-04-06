#!/usr/bin/env python3
import json
import cv2
import tensorflow as tf

class Akpd(object):
    def __init__(self, model_path, config_path, device='/gpu:0'):
        self.is_active = False
        self._model_path = str(model_path)
        self._config_path = config_path
        self._device_ix = device
        with config_path.open('r') as fp:
            self._config = json.load(fp)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(exc_type, exc_value, traceback)

    @staticmethod
    def _load_graph_def(model_path):
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_nodes=[n for n in graph_def.node]
            return graph_def, graph_nodes
        
    def open(self):
        if self.is_active:
            return
              
        self._device = tf.device(self._device_ix).__enter__()
        self._graph_def, self._graph_nodes = self._load_graph_def(self._model_path)
        assert self._graph_def, 'No graphdef created'
        assert len(self._graph_nodes) > 0,'No graph nodes found'
        self._graph = tf.Graph()
        self._session = tf.compat.v1.Session(graph=self._graph).__enter__()
        assert self._session, 'No session created'
        tf.compat.v1.disable_eager_execution()
        self._session.run(tf.compat.v1.global_variables_initializer())
        tf.graph_util.import_graph_def(self._graph_def, input_map=None, return_elements=None, name="", producer_op_list=None)
        self.is_active = True

    def close(self, exc_type=None, exc_value=None, traceback=None):
        if not self.is_active:
            return

        if self._session: 
            self._session.__exit__(exc_type, exc_value, traceback)
        if self._device:
            self._device.__exit__(exc_type, exc_value, traceback)
        self.is_active = False
        
    def _process(self, image):
        assert self.is_active, 'AKPD model not loaded'
        final_stage_heatmap = self._session.run(self._config['output_name'], feed_dict = {self._config['input_name']: image})
        hm = final_stage_heatmap.squeeze()
        return hm

    def process(self, left_image, right_image):
        left_hm = self._process(left_image)
        right_hm = self._process(right_image)
        return left_hm, right_hm