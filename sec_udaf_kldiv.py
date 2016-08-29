# -*-coding:UTF-8-*-#

"""
des: kl散度
create time: 2016-7-25 
version: 
author: weaponli
function:  
Copyright 1998 - 2108 TENCENT Inc. All Rights Reserved
modify:UserDefineAggregateFunction
    <author>        <modify time>       <des>
"""
import math
from base_udf import BaseUDAF

class SecUdafKLDiv(BaseUDAF):
    def __init__(self):
        BaseUDAF.__init__(self)
        self.dictBuffer = []
        
    def initialize(self):
        """初始化聚合函数lstBuffer,给定初始值
        """
        dictBuffer = {"first_num": [0.0],
                      "second_num": [0.0],
                      "first_list": [],
                      "second_list": [],
                      "zero_parameter": [2.220446049e-16]}
        return dictBuffer
    
    
    def update(self, dictBuffer, lstData):
        """处理每一行数据,如果某个值为空，那么去掉
           data[0] = 第一分布的值
           data[1] = 第2分布的值
           data[2] = 当第二分出出现0概率的时候，用什么值代替
        """
        try:
            floatFirstData = float(lstData[0])
            floatSecondData = float(lstData[1])
            floatZeroParam = float(lstData[2])
            
            #看看是否需要自动添加一点数值，不让概率为0
            floatEps = 2.220446049e-16
            if floatZeroParam >= 0:
                floatEps = 0.0
                
            if floatFirstData >= 0 and floatSecondData >= 0:
                floatFirstData += floatEps
                floatSecondData += floatEps
                dictBuffer["first_num"][0] += floatFirstData 
                dictBuffer["second_num"][0] += floatSecondData
                dictBuffer["first_list"].append(floatFirstData)
                dictBuffer["second_list"].append(floatSecondData)
                dictBuffer["zero_parameter"][0] = floatZeroParam
                
        except Exception, e:
            pass
        return dictBuffer
        
    def merge(self, buf, data):
        """局部聚合部分结果,返回buffer
         (暂不支持python UDAF的merge,后续版本可支持)
        >>> def merge(self, data):
        ...  self.buffer[0] += data[0]
        ...  self.buffer[1] += data[1]
        ...  return self.buffer
        """
        pass
        
        
    def __divergence(self, floatFirstPro, floatSecondPro, floatZeroParam):
        """返回不对称的数值"""
        if floatFirstPro == 0.0:
            return 0
        
        if floatSecondPro == 0.0:
            return floatZeroParam
            
        return floatFirstPro * math.log(floatFirstPro/floatSecondPro, 2)
        
        
    def eval(self, dictBuffer):
        """计算最终结果，返回结果值
        
        """
        nDataLen = len(dictBuffer["first_list"])
        if nDataLen < 1 or dictBuffer["first_num"][0] == 0 or dictBuffer["second_list"] == 0:
            return -1.0
        
        floatLeftKLDiv = 0.0
        floatRightKLDiv = 0.0
        floatSymKLDiv = 0.0
        floatJSKLDiv = 0.0
        
        lstRange = range(0, nDataLen)
        #print lstRange
        for nIndex in lstRange:
            #计算概率
            floatFirstPro = dictBuffer["first_list"][nIndex] / dictBuffer["first_num"][0]
            floatSecondPro = dictBuffer["second_list"][nIndex] / dictBuffer["second_num"][0]
            
            #左边的kl散度
            floatLeftDiv= self.__divergence(floatFirstPro, 
                                            floatSecondPro,
                                            dictBuffer["zero_parameter"][0])
            floatLeftKLDiv += floatLeftDiv
            
            #右边列的kl散度
            floatRightDiv= self.__divergence(floatSecondPro, 
                                             floatFirstPro,
                                             dictBuffer["zero_parameter"][0])
            floatRightKLDiv += floatRightDiv
            
            #对称kl散度
            floatSymKLDiv += ((floatLeftDiv + floatRightDiv)/ 2)
            
            #Jensen-Shannon divergence for a single sample (symmetric)
            floatMeanPro = (floatFirstPro + floatSecondPro) / 2
            floatJSKLDiv += ((self.__divergence(floatFirstPro, floatMeanPro, dictBuffer["zero_parameter"][0]) \
                              + self.__divergence(floatSecondPro, floatMeanPro, dictBuffer["zero_parameter"][0])) / 2) 
        
        return [floatLeftKLDiv, floatRightKLDiv, floatSymKLDiv, floatJSKLDiv]
        
        
        
        