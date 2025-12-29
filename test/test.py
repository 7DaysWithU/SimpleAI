from control import GeneralDispatch

# 生成总控实例
run = GeneralDispatch()

# 配置参数
run.configure('../config/settings.json')

# 训练所有模型并评估效果(非增量训练)
run.train(incremental = False)
run.eval()

# 加载模型到内存并预测x=10.0时的y值
# 示例数据关系: y = 2.5 * x + 1.0
run.load_model()
print(run.use([10.0]))
