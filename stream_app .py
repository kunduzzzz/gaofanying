import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(page_title="卵巢高反应预测模型", layout="wide")
st.title("卵巢高反应风险评估")
st.markdown("""
**临床用途**：本工具基于患者临床和实验室参数，预测控制性卵巢刺激过程中发生高反应的风险，辅助临床决策。
""")

# Load model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'xgboost_model.pkl')  # 请确保模型文件名为 ovarian_hyper_model.pkl
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Sidebar input features
with st.sidebar:
    st.header("患者参数")
    
    st.subheader("基础与内分泌指标")
    AMH = st.slider("AMH (ng/mL)", min_value=0.0, max_value=50.0, value=3.0, step=0.1,
                    help="抗苗勒管激素，反映卵巢储备功能")
    AFC = st.slider("AFC (个)", min_value=0, max_value=60, value=10, step=1,
                    help="窦卵泡计数，基础状态双侧卵巢2-10mm卵泡总数")
    FSH = st.slider("FSH (IU/L)", min_value=0.0, max_value=100.0, value=6.0, step=0.1,
                    help="基础促卵泡激素，月经周期第2-3天水平")
    age = st.slider("年龄 (岁)", min_value=20, max_value=45, value=30, step=1,
                    help="患者实际年龄")
    
    st.subheader("促排卵方案")
    Fangan = st.slider("方案编码 (Fangan)", min_value=0, max_value=4, value=2, step=1,
                       help="1=PPOS , 2=黄体期长方案, 3=GnRH 拮抗剂方案, 4=卵泡期长方案 , 5=短效 GnRH 激动剂长方案")

# Create input dataframe (特征顺序必须与训练时一致)
input_data = pd.DataFrame({
    'AMH': [AMH],
    'AFC': [AFC],
    'FSH': [FSH],
    'age': [age],
    'Fangan': [Fangan]
})

# Prediction and results
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("评估高反应风险"):
        # Probability prediction (假设模型正类为高反应)
        prob = model.predict_proba(input_data)[0][1]
        risk_level = "高风险" if prob >= 0.6 else "中风险" if prob >= 0.3 else "低风险"
        
        # Display results
        st.subheader("风险评估结果")
        st.metric(label="高反应预测概率", value=f"{prob:.1%}")
        
        st.markdown(f"""
        **风险等级**: {risk_level}
        
        **临床建议**:
        - 高风险: 考虑GnRH拮抗剂方案、减少起始促性腺激素剂量、考虑coasting或全胚冷冻
        - 中风险: 个体化启动剂量，密切监测卵泡发育和雌二醇水平
        - 低风险: 可按常规方案启动治疗
        """)

with col2:
    if 'prob' in locals():
        st.subheader("主要影响因素")
        st.markdown(f"""
        当前参数特征:
        - AMH: {AMH:.1f} ng/mL {'(显著升高)' if AMH > 5 else '(正常范围)'}
        - AFC: {AFC} 个 {'(显著增多)' if AFC > 20 else '(正常范围)'}
        - FSH: {FSH:.1f} IU/L {'(较低水平)' if FSH < 5 else '(正常范围)'}
        - 年龄: {age} 岁 {'(年轻)' if age < 35 else '(年龄偏大)'}
        - 方案编码: {Fangan} {'(长方案)' if Fangan==1 else '(拮抗剂)' if Fangan==2 else '(其他)'}
        """)
        
        # Visual indicator (相对贡献示意，非SHAP值)
        risk_factors = {
            'AMH': min(AMH/15, 1.0),      # AMH 常见高值上限15 ng/mL
            'AFC': min(AFC/35, 1.0),      # AFC 高值上限35个
            'FSH': max(0, min(1 - FSH/25, 1.0)), # FSH 越低风险越高，反向归一
            '年龄': max(0, min(1 - (age-20)/30, 1.0)), # 年龄越小风险越高
            'Fangan': (Fangan-1)/3        # 方案编码影响，1→0, 4→1
        }
        st.bar_chart(pd.DataFrame.from_dict(
            {'相对风险贡献': risk_factors}), 
            height=300
        )

# Guidelines and references
st.markdown("---")
st.info("""
**临床指南参考**:
- 预测高风险可考虑预防性使用GnRH拮抗剂、减少FSH剂量
- 获卵数>15个或E2峰值>3500 pg/mL提示高反应
- 建议进行卵巢过度刺激综合征（OHSS）预防管理
""")

# Disclaimer
st.warning("""
**免责声明**:
1. 本预测仅供临床决策参考，不可替代医生判断
2. 模型需经本地数据验证后使用
3. 最终治疗方案应结合患者个体情况制定
""")
