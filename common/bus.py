import os

from datetime import datetime as dt
from rknn.api import RKNN

from common.utils import check_recall,show_lg,show_er

class Bus():
    def __init__(self,mp,base_params,build_params,infer_params,test_params,export_params) -> None:
        self.mp = mp
        self.base_params = base_params
        self.build_params =build_params
        self.infer_params = infer_params
        self.test_params = test_params
        self.export_params = export_params
        self.results = os.path.join(os.getcwd(),"runs",dt.now().strftime('%Y_%m_%d-%H_%M_%S'))
        os.makedirs(self.results,exist_ok=True)
        self.rknn = RKNN(verbose=True,verbose_file=os.path.join(self.results,'runtime.log'))

    def base_info(self):
        show_lg(self.rknn.get_sdk_version())
        show_lg(self.rknn.list_devices())

    def init_env_rknn(self):
        if self.mp.endswith(".rknn"):
            show_lg('--> Loading model')
            ret = self.rknn.load_rknn(path=self.mp)
            check_recall(ret,"load model")
        else:
            if self.mp.endswith(".prototxt"):
                show_lg("--> Config model")
                self.rknn.config(mean_values=self.build_params.mean_v, std_values=self.build_params.std_v, quant_img_RGB2BGR=True ,\
                                 quantized_algorithm=self.build_params.quantized_a,quantized_method=self.build_params.quantized_m, \
                                    float_dtype=self.build_params.float_d,optimization_level=self.build_params.optimization_l, \
                                        )
                show_lg('--> Loading model')
                ret = self.rknn.load_caffe(model=self.mp, \
                                        blobs=self.build_params.export_info["additional_path"])
                check_recall(ret,"load model")
            else:
                show_lg("--> Config model")
                self.rknn.config(mean_values=self.build_params.mean_v, std_values=self.build_params.std_v)
                show_lg('--> Loading model')
                if self.mp.endswith(".tflite"):
                    ret = self.rknn.load_tflite(tf_pb=self.mp,
                                                    input_is_nchw=self.build_params.export_info["input_is_nchw"]
                                        )
                    check_recall(ret,"load model")
                elif self.mp.endswith(".pb"):
                    ret = self.rknn.load_tensorflow(tf_pb=self.mp,
                                                    inputs=self.build_params.export_info["inputname"]
                                        )
                    check_recall(ret,"load model")
                elif self.mp.endswith(".onnx"):
                    ret = self.rknn.load_onnx(model=self.mp,
                                                    inputs=self.build_params.export_info["inputname"],
                                                    input_size_list = self.build_params.export_info["input_size_list"],
                                                    input_initial_val= self.build_params.export_info["input_initial_val"],
                                                    outputs=self.build_params.export_info["outputs"])
                    check_recall(ret,"load model")

                elif self.mp.endswith(".cfg"):
                    ret = self.rknn.load_darknet(model=self.mp,
                                                weight=self.build_params.export_info["additional_path"]  
                                        )
                    check_recall(ret,"load model")

                elif self.mp.endswith(".pt"):
                    ret = self.rknn.load_pytorch(model=self.mp,
                                                    input_size_list=self.build_params.export_info["input_size_list"]
                                        )
                    check_recall(ret,"load model")

                else:
                    show_er(f"load model method not exist",ex=-1)
                show_lg('--> Building model')
                self.rknn.build(do_quantization=self.build_params.do_quantization ,
                                dataset=self.build_params.dataset,rknn_batch_size=self.build_params.rknn_batch_size)
                check_recall(ret,"build model")

    def export_rknn(self):
        show_lg('--> Export rknn model')
        self.export_rknn_path = self.mp[:self.mp.rfind(".")]+"_dist.rknn"
        ret = self.rknn.export_rknn(export_path=self.export_params.export_rknn_path,cpp_gen_cfg=self.export_params.cpp_gen_cfg)
        check_recall(ret,"export model")
    def runtime(self):
        ret = self.rknn.init_runtime(target=self.build_params.target,
                               device_id=self.build_params.device_id,
                               perf_debug=self.build_params.perf_debug,
                               eval_mem=self.build_params.eval_mem,
                               async_mode=self.build_params.async_mode,
                               core_mask=self.build_params.core_mask,
                               )
        check_recall(ret,"Initruntimeenvironment")

    def inference(self,data):
        res = self.rknn.inference(inputs=data,data_format=self.infer_params.data_format, \
                                  inputs_pass_through=self.infer_params.inputs_pass_through)
        return res

    def eval_performace(self):
        perf_result = self.rknn.eval_perf(is_print=self.test_params.is_print)
        with open(os.path.join(self.results,self.test_params.perf_f),'w') as f:
            f.write(" >>> Model performance analysis <<< \n")
            for info in perf_result:
                f.write(info+'\n')
            f.write("-"*20+'\n')
        perf_result = self.rknn.eval_memory(is_print=self.test_params.is_print)
        with open(os.path.join(self.results,self.test_params.perf_f),'w+') as f:
            f.write(" >>> Memory usage analysis <<< \n")
            for info in perf_result:
                f.write(info+'\n')
        ret = self.rknn.accuracy_analysis(inputs=self.test_params.inputs,\
                                                  output_dir=os.path.join(self.results,self.test_params.output_dir),\
                                                  target=self.test_params.target ,\
                                                  device_id=self.test_params.device_id)
        check_recall(ret,"accuracy_analysis")
    def release(self):
        self.rknn.release()