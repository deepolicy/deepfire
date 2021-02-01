# 导入自己设计的红蓝AI模型的类名.
# 注意命名智能体时, 要求红方智能体类名中包含Red字段,
# 蓝方智能体类名中包含Blue字段, 以便于被大规模对战调度程序自动区分和加载.
from .deepfire_red import RedAgent
from .deepfire_blue import BlueAgent