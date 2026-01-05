# -*- coding: utf-8 -*-

导入 日志
导入 系统
导入 操作系统
from  config  import  Config


def  logger ():
    # 获取引用者的文件名
    caller_file   =  sys._getframe ( 1 ) .f_code.co_filename​​​​
    file_name  =  os.path.splitext ( os.path.basename ( caller_file ) ) [ 0 ]​​​​​​
    #创建记录器
    logger  =  logging.getLogger ( file_name )​​
    日志记录器.设置级别(日志记录.调试)
    #创建一个文件内存，并明确指定utf-8编码
    file_handler  =  logging.FileHandler ( Config [ "log_path" ] + "/" + file_name + " .log" , encoding = 'utf-8 ' )     
    file_handler.setLevel ( logging.DEBUG )​​​​
    #格式设置并应用
    formatter  =  logging.Formatter ( '%(asctime)s - %(name)s - %(levelname)s - %(message) s ' )
    文件处理程序。setFormatter（格式化程序）
    logger.addHandler ( file_handler )​​
    返回 日志记录器


如果 __name__  ==  '__main__'：
    logger  =  logger ()
    记录器。info ( "logHandler初始化完成！！！" )
