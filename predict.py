import os
import joblib

# 加载模型
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
model = joblib.load(model_path)

def predict_scam(text):
    """输入文本，返回诈骗概率和关键词"""
    if not text.strip():
        return None
    
    # 预测概率 (0 到 1)
    prob = model.predict_proba([text])[0][1]
    
    # 提取让模型认为是诈骗的关键词（TF-IDF 权重最高的词）
    feature_names = model.named_steps['tfidfvectorizer'].get_feature_names_out()
    coefs = model.named_steps['multinomialnb'].feature_log_prob_[1]
    
    # 简单提取文本中命中的词
    top_terms = [word for word in feature_names if word in text.lower()][:3]
    
    return {
        "probability": prob,
        "top_terms": top_terms
    }

def to_finding(ml_result):
    """把模型结果转成主程序需要的格式"""
    if not ml_result or ml_result["probability"] < 0.5:
        return None
        
    return {
        "key": "ml_cue",
        "type": "🧠 ML Model Detection",
        "evidence": ", ".join(ml_result["top_terms"]) or "Pattern matching",
        "all_matches": ml_result["top_terms"],
        "explanation": f"The local ML model found this text has a {ml_result['probability']:.0%} probability of being a scam based on historical patterns.",
        "severity": "Medium" # 给 Medium，配合规则使用
    }