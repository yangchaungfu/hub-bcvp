# loader.py 中随机生成正负样本代码
    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        s1, s2, s3 = None, None, None
        #随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        p = random.choice(standard_question_index)
        #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        #随机负样本
        # else:
        s, n = random.sample(standard_question_index, 2)
        s3 = random.choice(self.knwb[n])
        if s1 is not None and s2 is not None and s3 is not None:
            return [s1, s2, s3]
        else:
            self.random_train_sample()

# model.py 中的模型训练部分forward的修改
#sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None):
        #同时传入两个句子
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)

# 最终模型预测结果
Connected to pydev debugger (build 241.17890.14)
2025-12-10 21:02:41,377 - __main__ - INFO - gpu可以使用，迁移模型至gpu
2025-12-10 21:02:57,703 - __main__ - INFO - epoch 1 begin
2025-12-10 21:03:03,955 - __main__ - INFO - epoch average loss: 0.072247
2025-12-10 21:03:03,955 - __main__ - INFO - 开始测试第1轮模型效果：
2025-12-10 21:03:04,023 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,023 - __main__ - INFO - 预测正确条目：396，预测错误条目：68
2025-12-10 21:03:04,023 - __main__ - INFO - 预测准确率：0.853448
2025-12-10 21:03:04,023 - __main__ - INFO - --------------------
2025-12-10 21:03:04,023 - __main__ - INFO - epoch 2 begin
2025-12-10 21:03:04,066 - __main__ - INFO - epoch average loss: 0.065377
2025-12-10 21:03:04,066 - __main__ - INFO - 开始测试第2轮模型效果：
2025-12-10 21:03:04,163 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,163 - __main__ - INFO - 预测正确条目：393，预测错误条目：71
2025-12-10 21:03:04,163 - __main__ - INFO - 预测准确率：0.846983
2025-12-10 21:03:04,163 - __main__ - INFO - --------------------
2025-12-10 21:03:04,163 - __main__ - INFO - epoch 3 begin
2025-12-10 21:03:04,222 - __main__ - INFO - epoch average loss: 0.064301
2025-12-10 21:03:04,222 - __main__ - INFO - 开始测试第3轮模型效果：
2025-12-10 21:03:04,303 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,303 - __main__ - INFO - 预测正确条目：398，预测错误条目：66
2025-12-10 21:03:04,303 - __main__ - INFO - 预测准确率：0.857759
2025-12-10 21:03:04,303 - __main__ - INFO - --------------------
2025-12-10 21:03:04,303 - __main__ - INFO - epoch 4 begin
2025-12-10 21:03:04,323 - __main__ - INFO - epoch average loss: 0.062797
2025-12-10 21:03:04,323 - __main__ - INFO - 开始测试第4轮模型效果：
2025-12-10 21:03:04,397 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,397 - __main__ - INFO - 预测正确条目：406，预测错误条目：58
2025-12-10 21:03:04,397 - __main__ - INFO - 预测准确率：0.875000
2025-12-10 21:03:04,397 - __main__ - INFO - --------------------
2025-12-10 21:03:04,397 - __main__ - INFO - epoch 5 begin
2025-12-10 21:03:04,423 - __main__ - INFO - epoch average loss: 0.059349
2025-12-10 21:03:04,423 - __main__ - INFO - 开始测试第5轮模型效果：
2025-12-10 21:03:04,507 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,507 - __main__ - INFO - 预测正确条目：414，预测错误条目：50
2025-12-10 21:03:04,507 - __main__ - INFO - 预测准确率：0.892241
2025-12-10 21:03:04,507 - __main__ - INFO - --------------------
2025-12-10 21:03:04,507 - __main__ - INFO - epoch 6 begin
2025-12-10 21:03:04,533 - __main__ - INFO - epoch average loss: 0.063350
2025-12-10 21:03:04,533 - __main__ - INFO - 开始测试第6轮模型效果：
2025-12-10 21:03:04,602 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,602 - __main__ - INFO - 预测正确条目：413，预测错误条目：51
2025-12-10 21:03:04,602 - __main__ - INFO - 预测准确率：0.890086
2025-12-10 21:03:04,602 - __main__ - INFO - --------------------
2025-12-10 21:03:04,602 - __main__ - INFO - epoch 7 begin
2025-12-10 21:03:04,632 - __main__ - INFO - epoch average loss: 0.059899
2025-12-10 21:03:04,632 - __main__ - INFO - 开始测试第7轮模型效果：
2025-12-10 21:03:04,703 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,703 - __main__ - INFO - 预测正确条目：417，预测错误条目：47
2025-12-10 21:03:04,703 - __main__ - INFO - 预测准确率：0.898707
2025-12-10 21:03:04,703 - __main__ - INFO - --------------------
2025-12-10 21:03:04,703 - __main__ - INFO - epoch 8 begin
2025-12-10 21:03:04,736 - __main__ - INFO - epoch average loss: 0.070451
2025-12-10 21:03:04,736 - __main__ - INFO - 开始测试第8轮模型效果：
2025-12-10 21:03:04,803 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,803 - __main__ - INFO - 预测正确条目：417，预测错误条目：47
2025-12-10 21:03:04,803 - __main__ - INFO - 预测准确率：0.898707
2025-12-10 21:03:04,803 - __main__ - INFO - --------------------
2025-12-10 21:03:04,803 - __main__ - INFO - epoch 9 begin
2025-12-10 21:03:04,829 - __main__ - INFO - epoch average loss: 0.082125
2025-12-10 21:03:04,829 - __main__ - INFO - 开始测试第9轮模型效果：
2025-12-10 21:03:04,920 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:04,920 - __main__ - INFO - 预测正确条目：421，预测错误条目：43
2025-12-10 21:03:04,920 - __main__ - INFO - 预测准确率：0.907328
2025-12-10 21:03:04,920 - __main__ - INFO - --------------------
2025-12-10 21:03:04,920 - __main__ - INFO - epoch 10 begin
2025-12-10 21:03:04,951 - __main__ - INFO - epoch average loss: nan
2025-12-10 21:03:04,951 - __main__ - INFO - 开始测试第10轮模型效果：
2025-12-10 21:03:05,021 - __main__ - INFO - 预测集合条目总量：464
2025-12-10 21:03:05,021 - __main__ - INFO - 预测正确条目：422，预测错误条目：42
2025-12-10 21:03:05,021 - __main__ - INFO - 预测准确率：0.909483
2025-12-10 21:03:05,021 - __main__ - INFO - --------------------
