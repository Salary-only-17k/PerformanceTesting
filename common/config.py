import pathlib
from datetime import datetime
from rknn.api import RKNN
class base_params():
    def __init__(self):
        # [load model]
        self.quant_img =False
        self.img_pth = ""
        self.hw = (224,224)

class build_params():
    def __init__(self):
        #! [load model]
        self.export_info = {"additional_path":"weights/resnet50v2.cfg",
                       "inputname":["input1","input2"],"inputs":None,
                       "input_size_list":[[640,640]],
                       "input_initial_val":None,
                       "outputs":None,
                       "input_is_nchw":"NHWC"}
      
        #! [config]
        # mean_values：输入的均值.[[128,128,128]]
        self.mean_v =[[123.68, 116.28, 103.53]]

        # std_values：输入的均值.[[128,128,128]]
        self.std_v = [[57.38, 57.38, 57.38]]

        # quantized_dtype：量化类型，目前支持的量化类型有asymmetric_quantized-8
        # quantized_d = asymmetric_quantized-8

        # quantized_algorithm：计算每一层的量化参数时采用的量化算法 normal，mmse及kl_divergence。默认值为normal。
        self.quantized_a = "normal"

        # quantized_method：目前支持layer或者channel。默认值为channel。
        self.quantized_m = "channel" 

        # float_dtype:用于指定非量化情况下的浮点的数据类型,目前支持的数据类型有float16。
        self.float_d = "float16"

        # optimization_level：模型优化等级。默认值为3。通过修改模型优化等级，可以关掉部分或全部模型转换过程中使用到的优化规则。该参数的默认值为3，打开所有优化选项。值为2或1时关闭一部分可能会对部分模型精度产生影响的优化选项，值为0时关闭所有优化选项
        self.optimization_l=3

        # target_platform：指定RKNN模型是基于哪个目标芯片平台生成的。目前支持“rk3566”、“rk3568”、“rk3588”、“rv1103”、“rv1106”和“rk3562”。该参数对大小写不敏感。默认值为None，表示默认为rk3566。
        self.target_p ="rk3588"
        # custom_stirng：添加自定义字符串信息到RKNN模型，可以在runtime时通过query查询到该信息，方便部署时根据不同的RKNN模型做特殊的处理。默认值为None
        self.custom_s =None
        # remove_weight：去除conv等权重以生成一个RKNN的从模型
        self.remove_w= False
        # compress_weight：压缩模型权重，可以减小RKNN模型的大小。默认值为False。
        self.compress_w = False
        # single_core_mode：是否仅生成单核模型，可以减小RKNN模型的大小和内存消耗。默认值为False。目前仅对RK3588生效。默认值为False。
        self.single_c_m = False
        # model_pruning：对模型进行无损剪枝。
        self.model_p = False
        # op_target
        self.op_t=None
        # dynamic_input 动态输入，只支持动态输入模型 dynamic_input=[[[1,3,224,224]],[[1,3,192,192]],[[1,3160,160]]]
        self.d_input = None

        #! [build]
        self.rknn_batch_size=4         # input batchsize
        self.do_quantization=True      # 是否量化
        self.dataset = "data/dataset.txt"   # 量化矫正数据
        #! [runtime]
        # 目标硬件平台，支持“rk3566”、“rk3568”、“rk3588”、“rv1103”、“rv1106”、“rk3562”。默认值为None，即在PC使用工具时，模型在模拟器上运行。
        self.target = None
        # device_id：设备编号，如果PC连接多台设备时，需要指定该参数，设备编号可以通过“list_devices”接口查看。默认值为None。
        self.device_id =None
        # perf_debug：进行性能评估时是否开启debug模式。在debug模式下，可以获取到每一层的运行时间，否则只能获取模型运行的总时间。默认值为False。
        self.perf_debug = False
        # eval_mem：是否进入内存评估模式。进入内存评估模式后，可以调用eval_memory接口获取模型运行时的内存使用情况。默认值为False。
        self.eval_mem = False
        # async_mode：是否使用异步模式。默认值为False。调用推理接口时，涉及设置输入图片、模型推理、获取推理结果三个阶段。如果开启了异步模式，设置当前帧的输入将与推理上一帧同时进行，所以除第一帧外，之后的每一帧都可以隐藏设置输入的时间，从而提升性能。在异步模式下，每次返回的推理结果都是上一帧的。（目前版本该参数暂不支持）
        # （目前版本该参数暂不支持）
        # self.async_mode = False
        
        # core_mask：设置运行时的NPU核心。支持的平台为RK3588，支持的配置如下：
        # 0 RKNN.NPU_CORE_AUTO：表示自动调度模型，自动运行在当前空闲的NPU核上。
        # 1 RKNN.NPU_CORE_0：表示运行在NPU0核心上。
        # 2 RKNN.NPU_CORE_1：表示运行在NPU1核心上。
        # 3 RKNN.NPU_CORE_2：表示运行在NPU2核心上。
        # 4 RKNN.NPU_CORE_0_1：表示同时运行在NPU0、NPU1核心上。
        # 5 RKNN.NPU_CORE_0_1_2：表示同时运行在NPU0、NPU1、NPU2核心上。默认值为RKNN.NPU_CORE_AUTO。
        self.core_mask = self.choose_core()[0]

    def choose_core(self):
        core = {0:RKNN.NPU_CORE_AUTO,
                1:RKNN.NPU_CORE_0,
                2:RKNN.NPU_CORE_1,
                3:RKNN.NPU_CORE_0_1,
                4: RKNN.NPU_CORE_2,
                5: RKNN.NPU_CORE_0_1,
                6: RKNN.NPU_CORE_0_1_2
                }
        return core
       

      

class infer_params():
    def __init__(self) -> None:
        # [inferance]
        # data_format：输入数据的layout列表，“nchw”或“nhwc”，只对4维的输入有效。默认值为None，表示所有输入的layout都为NHWC。
        self.data_format = "nhwc"
        # inputs_pass_through：输入的透传列表。默认值为None，表示所有输入都不透传。
        # 非透传模式下，在将输入传给NPU驱动之前，工具会对输入进行减均值、除方差等操作；
        # 而透传模式下，不会做这些操作，而是直接将输入传给NPU。该参数的值是一个列表，
        # 比如要透传input0，不透传input1，则该参数的值为[1,0]。
        self.inputs_pass_through = [1]


class test_params():
    def __init__(self) -> None:
        # ![eval_perf]
        self.is_print = True
        # result performace save filename
        # ![eval_memory]
        self.perf_f = "perf_info.txt"
        #! [accuracy_analysis]
        self.inputs= img_pth_lst("path/to/dir")
        # output_dir：输出目录
        self.output_dir = 'snapshot'
        # target：目标硬件平台
        self.target = None
        # device_id：设备编号
        self.device_id = None


class export_params():
    def __init__(self) -> None:
        # [export]
        self.export_rknn_path = "model.rknn"
        self.cpp_gen_cfg = True

def img_pth_lst(dir):
    tmp = []
    for fmt in [".jpg",".png",".bmp",".npy"]:
        tmp += list(pathlib.Path(dir).glob(f"**/*{fmt}"))
    return [str(t) for t in tmp]