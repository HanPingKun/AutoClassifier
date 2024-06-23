from flask import Flask, request, render_template, redirect, url_for
import os
import json
import copy
import pandas as pd
import numpy as np
import torch
import logging
import random
import pkg_resources
import sklearn
from rxnfp.models import SmilesClassificationModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from dotenv import load_dotenv, find_dotenv

# 创建一个使用Flask框架的Web应用程序，并配置一些基本设置。
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# 点击网页预测按钮进行预测
def predict():
    # 传入新加数据
    rxn_class = request.form['rxn_class']
    class_id = int(request.form['class_id'])
    class_name = request.form['class_name']
    file = request.files['file']
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # 创建一个日志记录器，并加载 .env 文件中的环境变量到系统环境变量中。
        os.environ["WANDB_API_KEY"] = "d99eba8f44eeef348acf2b6712732b73b473cc45"
        os.environ["WANDB_MODE"] = "offline"
        logger.info("Loading environment variables")
        load_dotenv(find_dotenv())

        # 加载反应类别与其对应的 ID 和名称的映射关系，并提取所有反应类别，按字典序排序
        with open('../data/rxnclass2id.json', 'r') as f:
            rxnclass2id = json.load(f)
        with open('../data/rxnclass2name.json', 'r') as f:
            rxnclass2name = json.load(f)

        # 修改 JSON 对象，添加反应类型
        rxnclass2id[rxn_class] = class_id
        rxnclass2name[rxn_class] = class_name

        # 将修改后的 JSON 对象保存回文件
        with open('../data/rxnclass2id.json', 'w') as file:
            json.dump(rxnclass2id, file, indent=4)
        with open('../data/rxnclass2name.json', 'w') as file:
            json.dump(rxnclass2name, file, indent=4)
        
        all_classes = sorted(rxnclass2id.keys())

        # 从 Excel 文件中读取反应数据，并根据之前加载的 rxnclass2id 字典，为每个反应类别添加相应的 class_id
        schneider_df = pd.read_excel('../data/schneider50k.xlsx', index_col=0)
        schneider_df['class_id'] = [rxnclass2id[c] for c in schneider_df.rxn_class]
        train_df_initial = schneider_df[schneider_df.split == 'train']
        test_df = schneider_df[schneider_df.split == 'test']

        # 划分训练集和验证集
        train_df2 = pd.DataFrame()
        eval_df = pd.DataFrame()
        for class_id, group in train_df_initial.groupby('class_id'):
            train, val = train_test_split(group, test_size=1/20, random_state=42)
            train_df2 = pd.concat([train_df2, train])
            eval_df = pd.concat([eval_df, val])

        # 添加新数据
        new_schneider_df = pd.read_excel(file_path, index_col=0)
        new_train_df = new_schneider_df[new_schneider_df.split == 'train']
        new_test_df = new_schneider_df[new_schneider_df.split == 'test']

        # 划分新训练集和验证集并与原数据集合并
        new_train_df2, new_eval_df = train_test_split(new_schneider_df, test_size=1/20, random_state=42)
        train_df2 = pd.concat([train_df2, new_train_df2])
        eval_df = pd.concat([eval_df[['rxn', 'class_id']], new_eval_df[['rxn', 'class_id']]])

        # 将 eval_df 数据框的列名重命名为 text 和 class_id
        eval_df.columns = ['text', 'class_id']
        all_train_reactions = train_df2.rxn.values.tolist()
        corresponding_labels = train_df2.class_id.values.tolist()
        final_train_df = pd.DataFrame({'text': all_train_reactions, 'class_id': corresponding_labels})
        final_train_df = final_train_df.sample(frac=1., random_state=42)

        # 开始训练
        model_args = {
            'wandb_project': 'nmi_uspto_1000_class+51k', 'num_train_epochs': 10, 'overwrite_output_dir': True,
            'learning_rate': 2e-5, 'gradient_accumulation_steps': 1,
            'regression': False, "num_labels": len(final_train_df.class_id.unique()), "fp16": False,
            "evaluate_during_training": True, 'manual_seed': 42,
            "max_seq_length": 512, "train_batch_size": 1, "warmup_ratio": 0.00,
            'output_dir': 'out/bert_class_1k_tpl+(50+1)k',
            'thread_count': 8,
            'multiprocessing_chunksize':16,
        }
        model_path = '../out/bert_mlm_1k_tpl'
        model = SmilesClassificationModel("bert", model_path, num_labels=len(final_train_df.class_id.unique()), args=model_args, use_cuda=torch.cuda.is_available())
        # 启用缓存
        model.train_model(final_train_df, eval_df=eval_df, acc=metrics.accuracy_score, mcc=metrics.matthews_corrcoef)
        logger.info("model training over")

        # 调用训练好的模型
        train_model_path = '../out/bert_class_1k_tpl+(50+1)k'
        model = SmilesClassificationModel("bert", train_model_path, use_cuda=torch.cuda.is_available())

       #读入test数据集(原始)
        eval_df_f = test_df[['rxn']]
        class_id = test_df[['class_id']]
        data_to_predict = eval_df_f['rxn'].values.tolist()
        #读入test数据集(新加)
        eval_df2_f = new_test_df[['rxn']]
        class_id2 = new_test_df[['class_id']]
        data_to_predict2 = eval_df2_f['rxn'].values.tolist()
        #预测（原始）
        y_preds = model.predict(data_to_predict)
        df_preds = pd.DataFrame(y_preds)
        #预测（新加）
        y_preds2 = model.predict(data_to_predict2)
        df_preds2 = pd.DataFrame(y_preds2)
        #结果处理(原始)
        result = df_preds.iloc[[0]]
        result_transposed = result.transpose()
        result_transposed.reset_index(inplace=True)
        result_transposed.to_csv('../out/data/test50k.csv', index=False)
        #结果处理(新加)
        result2 = df_preds2.iloc[[0]]
        result_transposed2 = result2.transpose()
        result_transposed2.reset_index(inplace=True)
        result_transposed2.to_csv('../out/data/newtest50k.csv', index=False)
        #真实值记录
        class_id.to_csv('../out/data/true50k.csv', index=False)
        class_id2.to_csv('../out/data/newtrue50k.csv', index=False)
        #准确率（原始）
        true_values = pd.read_csv('../out/data/true50k.csv')
        predicted_values = pd.read_csv('../out/data/test50k.csv')
        predicted_values = predicted_values.iloc[:, 1]
        true_values = true_values.iloc[0:, 0]
        accuracy = (predicted_values == true_values).mean()
        print("原始数据预测正确率为")
        print(accuracy)
        #准确率（新加）
        true_values2 = pd.read_csv('../out/data/newtrue50k.csv')
        predicted_values2 = pd.read_csv('../out/data/newtest50k.csv')
        predicted_values2 = predicted_values2.iloc[:, 1]
        true_values2 = true_values2.iloc[0:, 0]
        accuracy2 = (predicted_values2 == true_values2).mean()
        print("新加数据预测正确率为")
        print(accuracy2)
        
        return f"原始数据预测正确率为: {accuracy}, 新数据预测正确率为: {accuracy2}"

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
