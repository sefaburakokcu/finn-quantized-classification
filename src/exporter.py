# Base code is obtained from https://github.com/QDucasse/nn_benchmark/blob/master/nn_benchmark/core/exporter.py

import torch
import brevitas.onnx as bo

class Exporter(object):

    def export_onnx(self, model, output_dir_path, act_bit_width=4, weight_bit_width=4, 
                    input_bit_width=8, input_shape=[1, 3, 32, 32], epoch=10, input_tensor=None, quantized=True):
        '''Export the model in ONNX format. A different export is provided is the
           network is a quantized one because the quantizations need to be stored as
           specific ONNX attributes'''
        model = model.to('cpu')
        if quantized:
            self.quant_export(model=model, output_dir_path=output_dir_path,
                              act_bit_width=act_bit_width, weight_bit_width=weight_bit_width,
                              input_bit_width=input_bit_width, input_shape=input_shape, epoch=epoch,
                              input_tensor=input_tensor)
        else:
            self.base_export(model=model, output_dir_path=output_dir_path, input_shape=input_shape)


    def base_export(self, model, output_dir_path, input_shape=[1, 3, 32, 32]):
        input = torch.ones(input_shape, dtype=torch.float32)
        torch.onnx.export(model, input, output_dir_path +"/"+ model.name + ".onnx")

    def quant_export(self, model, output_dir_path,
                     act_bit_width=4, weight_bit_width=4,
                     input_bit_width=8, epoch = 10, input_shape=[1, 3, 32, 32],
                     input_tensor=None, torch_onnx_kwargs={}):
        model_act_bit_width    = "_A" + str(act_bit_width)
        model_weight_bit_width = "W" + str(weight_bit_width)
        model_input_bit_width  = "I" + str(input_bit_width)
        model_epoch            = "_E" + str(epoch)
        model_name_with_attr   = "".join([model.name, model_act_bit_width, model_weight_bit_width, model_input_bit_width, model_epoch])
        bo.export_finn_onnx(module=model,
                            input_shape=input_shape,
                            export_path=output_dir_path +"/"+ model_name_with_attr + ".onnx",
                            input_t=input_tensor)

