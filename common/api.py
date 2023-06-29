import abc
import cv2
import numpy as np
from common.bus import Bus
from common.utils import show_lg
from common.config import base_params,build_params,infer_params,test_params,export_params

class stream(metaclass=abc.ABCMeta):
    def __init__(self,mode,mp) -> None:
        self.base = Bus(mp,base_params,build_params,infer_params,test_params,export_params)
        self.mode = mode
        self.mp = mp
        self.base_params = base_params
        show_lg(base_params.__dict__)
        self.build_params =build_params
        show_lg(build_params.__dict__)
        self.infer_params = infer_params
        show_lg(infer_params.__dict__)
        self.test_params = test_params
        show_lg(test_params.__dict__)
        self.export_params = export_params
        show_lg(export_params.__dict__)
    def precope(self,*args):
        data = []
        for _ in range(self.build_params.rknn_batch_size):
            img = cv2.imread(self.base_params.img_pth)
            img = cv2.resize(img,(self.base_params.hw[0],self.base_params.hw[1]))
            data.append(img)
        return data
    @abc.abstractmethod
    def post(self,*args):
        ...
    
    def env_info(self):
        self.base.base_info()
    
    def infer(self):
        self.base.init_env_rknn()
        data = self.precope()
        self.base.runtime(data)
        out = self.base.inference()
        self.post(out)
        # print(res)
        # return  res
    def test(self):
        self.base.init_env_rknn()
        self.base.runtime()
        self.base.eval_performace()
        self.base.release()
        
    def export(self):
        self.base.init_env_rknn()
        self.base.export_rknn()
        self.base.release()
        
    
        
class cls_stream(stream):
    def __init__(self,mp) -> None:
        super().__init__(mp)
        
    def post(self,outputs):
        output_ = outputs[0].reshape((-1, 1000))
        for output in output_:
            output_sorted = sorted(output, reverse=True)
            top5_str = 'mobilenet_v1\n-----TOP 5-----\n'
            for i in range(5):
                value = output_sorted[i]
                index = np.where(output == value)
                for j in range(len(index)):
                    if (i + j) >= 5:
                        break
                    if value > 0:
                        topi = '{}: {}\n'.format(index[j], value)
                    else:
                        topi = '-1: 0.0\n'
                    top5_str += topi
            show_lg(top5_str)

class det_stream(stream):
    def __init__(self,mp) -> None:
        super().__init__(mp)
        
    def post(self,outputs):
        print("Not realized")
        exit(-1)    
    
class seg_stream(stream):
    def __init__(self,mp) -> None:
        super().__init__(mp)
        
    def post(self,outputs):
        print("Not realized")
        exit(-1)    


class otr_stream(stream):
    def __init__(self,mp) -> None:
        super().__init__(mp)
        
    def post(self,outputs):
        print("Not realized")
        exit(-1)    