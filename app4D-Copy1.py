import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


# 加载模型
model1 = load_model('fev1_fvc_prediction_model')
model2 = load_model('fev1pred prediction_model') 
model3 = load_model('GOLDCOPD_prediction_model') 

# 定义一个函数用于预测
def predict(input_data):
    predictions1 = predict_model(model1, data=input_data)
    predictions2 = predict_model(model2, data=input_data)
    predictions3 = predict_model(model3, data=input_data)
    return predictions1, predictions2, predictions3


# 标题和描述
st.title('COPD prediction model / 赛博算命——测测你的肺功能怎么样')
st.write("""
## 请填写 Input
请在左侧栏输入参数值，然后点击“预测”按钮 Enter the parameter values in the left column and click the "Predict" button。
""")

# 创建输入数据表单
st.header('---')

def user_input_features():
    # 定义显示标签和对应的原始参数值的映射
    options1 = {'男male': 0, '女female': 1}
    options2 = {'40-49岁/years': 1, '50-59岁year': 2, '60-69岁year': 3, '>70岁years': 4}
    options3 = {'<18.5kg/m2': 7, '18.6-23.9 kg/m2': 4, '24.0-27.9kg/m2':1, '>28.0kg/m2': 0}
    options4 = {'否False': 0, '是Ture': 1}
    options5 = {'否False': 0, '是Ture': 1}
    options6 = {'仅剧烈活动后气促Shortness of breath after strenuous activity only': 1, '平地快走或爬坡时气促Shortness of breath when walking fast or climbing on flat ground': 2, '活动时需要频繁休息，爬2层楼也感气促Activities need frequent rest, climbing 2 floors also feel shortness of breath': 3}
    options7 = {'否False': 0, '是/Ture': 1}
    options8 = {'从不吸烟Never smoking': 1, '1-14.9包·年pack·year': 2, '15-29.9包·年pack·year': 3, '≥30包·年pack·year': 4}
    options9 = {'否False': 0, '是Ture': 1}
    
    # 选择框，显示用户友好的标签
    selected_option1 = st.sidebar.selectbox('性别Sex', list(options1.keys()))
    selected_option2 = st.sidebar.selectbox('年龄Age', list(options2.keys()))
    selected_option3 = st.sidebar.selectbox('体重指数BMI=体重kg/身高*身高m', list(options3.keys()))
    selected_option4 = st.sidebar.selectbox('长期咳嗽或咳痰Cough or phlegm', list(options4.keys()))
    selected_option5 = st.sidebar.selectbox('反复发生的喘息Wheeze', list(options5.keys()))
    selected_option6 = st.sidebar.selectbox('活动后气促mMRC Dyspnea index', list(options6.keys()))
    selected_option7 = st.sidebar.selectbox('曾诊断为肺气肿Emphysema history', list(options7.keys()))
    selected_option8 = st.sidebar.selectbox('吸烟指数Smoking index 每天吸烟几包x吸烟几年', list(options8.keys()))
    selected_option9 = st.sidebar.selectbox('过去一年中是否使用呼吸药物治疗/Drug use history of respiratory diseases', list(options9.keys()))
    
    # 获取原始参数值
    param1 = options1[selected_option1]
    param2 = options2[selected_option2]
    param3 = options3[selected_option3]
    param4 = options4[selected_option4]
    param5 = options5[selected_option5]
    param6 = options6[selected_option6]
    param7 = options7[selected_option7]
    param8 = options8[selected_option8]
    param9 = options9[selected_option9]

    # 根据你的模型输入特征添加更多的参数
    data = {'S': param1, 'A': param2, 'B': param3, 'CP': param4, 
            'Wh': param5, 'mMRC3': param6, 'LD12': param7, 'Sidx': param8, 'Drugu': param9 }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# 显示输入参数
st.subheader('模型内部参数')
st.write(input_df)

# 做预测
if st.button('点击进行预测'):
    output1, output2, output3 = predict(input_df)
    
    # 显示预测结果的结构
    st.subheader('FEV1/FVC 预测结果')
    st.write(output1)
    st.subheader('FEV1%pred 预测结果')
    st.write(output2)
    st.subheader('是否为慢阻肺预测结果')
    st.write(output3)

    # 访问预测结果的标签
    try:
        fev1_fvc_score = output1['prediction_label'].values[0]  # 获取 FEV1/FVC 的预测值
        fev1_pred_score = output2['prediction_label'].values[0]  # 获取 FEV1%pred 的预测值
        GOLDCOPD_score = output3['prediction_label'].values[0]  # 获取 GOLDCOPD 的预测值

        st.subheader('预测结果(%)')
        st.write(f"您目前的 FEV1/FVC 预测值是: {fev1_fvc_score}")
        st.write(f"您目前的 FEV1%pred 预测值是: {fev1_pred_score}")
        st.write(f"您目前的 GOLDCOPD 预测值是: {GOLDCOPD_score}")

       # 仅当 GOLDCOPD_score == 1 时，输出慢阻肺警告信息并忽略其他结果
    if GOLDCOPD_score == 1:
        st.warning("您目前很可能患有慢阻肺，请进一步行肺功能检查。")
    else:
        # GOLDCOPD_score 为 0 时，显示进一步结果
      if fev1_fvc_score < 72 and fev1_pred_score < 78:
            st.error("您可能患有中度及以上慢阻肺，请立即联系呼吸专科医生。")
      elif 72 <= fev1_fvc_score < 80:
            st.warning("您目前还不是慢阻肺，但有患上慢阻肺的风险，请戒烟，增加体重，加强锻炼，参加肺功能筛查测试或纳入年度体检计划。")
      elif fev1_fvc_score >= 72 and fev1_pred_score <= 78:
            st.warning("您可能存在保留比值肺功能受损，请关注您的呼吸健康情况，建议进一步行肺功能筛查测试。")
      elif fev1_fvc_score <= 69:
            st.warning("您可能存在阻塞性通气功能障碍，请关注您的呼吸健康情况，建议进一步行肺功能筛查测试。")
      else:
            st.success("您目前不太可能患有慢阻肺。")
            st.error("无法识别的 GOLDCOPD 评分。")
            
    except KeyError as e:
        st.error(f"发生错误: 找不到预测结果列 {e}")
    except IndexError:
        st.error("发生错误: 预测结果没有返回值。请检查模型输出。")
    except Exception as e:
        st.error(f"发生错误: {e}")
       
