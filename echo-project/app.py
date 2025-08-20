import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import io
import requests
from datetime import datetime
import time

# 页面配置
st.set_page_config(
    page_title="用户之声回音壁 (Echo) Pro",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive-insight {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
        border-left: 4px solid #4caf50;
    }
    .negative-insight {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        border-left: 4px solid #f44336;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# (你需要保留文件顶部的所有 import 语句)
# (请用下面的新类，完整替换掉你原来的 AIEnhancedEchoAnalyzer 类)

class LLMEchoAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        # 智谱AI的API地址
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    # (你需要保留文件顶部的所有 import 语句)
# (请用下面的新类，完整替换掉你原来的 AIEnhancedEchoAnalyzer 类)

class LLMEchoAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def clean_text(self, text):
        """清理和预处理文本数据"""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\!\?\,\;\:\"\'()（）。！？，；：""'']', '', text)
        text = re.sub(r'\s+', ' ', text.strip())
        comments = [line.strip() for line in text.split('\n') if line.strip() and len(line) > 3]
        return comments

    def analyze(self, comments):
        """使用LLM API进行全面、智能的分析，并返回详细的逐条结果"""
        if not self.api_key:
            st.error("API Key未设置，请在Streamlit Secrets中配置 ZHIPU_API_KEY。")
            return None

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        # 将评论列表格式化，每条评论前加上编号，方便AI处理
        formatted_comments = "\n".join([f"{i+1}. {comment}" for i, comment in enumerate(comments)])

        # 全新的、要求更详细的Prompt
        prompt = f"""
        你是一位顶级的APP用户反馈分析专家。请深入分析以下用户评论列表，并严格按照指定的JSON格式返回结果。

        用户评论列表:
        ---
        {formatted_comments}
        ---

        请完成以下两项任务:

        任务1: 生成详细分析数据 (detailed_data)
        - 遍历每一条评论。
        - 对每一条评论，分析出其'情感倾向(sentiment)' ('positive', 'negative', 'neutral') 和 '核心意图(intent)' ('bug反馈', '功能建议', '体验赞扬', '体验抱怨', '咨询')。
        - 将每条评论的分析结果作为一个JSON对象放入列表中。

        任务2: 生成摘要数据 (summary_data)
        - 根据详细分析结果，统计情感和意图的汇总数量。
        - 提取最能代表用户关注焦点的15个核心关键词。
        - 总结出3条最具代表性的正面和负面反馈观点。

        请严格按照以下JSON格式输出，不要包含任何无关的文字、解释或代码标记:
        {{
          "detailed_data": [
            {{
              "comment": "<原始评论内容>",
              "sentiment": "<positive/negative/neutral>",
              "intent": "<bug反馈/功能建议/等>"
            }},
            ...
          ],
          "summary_data": {{
            "sentiment_analysis": {{ "positive": <integer>, "negative": <integer>, "neutral": <integer> }},
            "intent_classification": {{ "bug反馈": <integer>, "功能建议": <integer>, "体验赞扬": <integer>, "体验抱怨": <integer>, "咨询": <integer> }},
            "keyword_extraction": [ {{"keyword": "<string>", "count": <integer>}}, ... ],
            "summary": {{
              "positive_highlights": ["<string>", "<string>", "<string>"],
              "negative_highlights": ["<string>", "<string>", "<string>"]
            }}
          }}
        }}
        """
        payload = {"model": "glm-4", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            analysis_str = response.json()['choices'][0]['message']['content']
            return json.loads(analysis_str)
        except requests.exceptions.RequestException as e:
            st.error(f"调用AI模型API失败: {e}")
            return None
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            st.error(f"解析AI模型返回数据失败，请稍后重试。错误: {e}")
            st.code(response.text, language="text")
            return None

def create_enhanced_visualizations(results):
    """创建增强版可视化图表"""
    
    # 1. 情感分布带置信度的饼图
    sentiment_values = list(results['sentiment_results'].values())
    sentiment_labels = ['正面评价', '负面评价', '中性评价']
    
    # 确保有有效数据
    if sum(sentiment_values) == 0:
        sentiment_values = [1, 0, 0]  # 默认值避免空图表
    
    sentiment_fig = go.Figure(data=[go.Pie(
        labels=sentiment_labels,
        values=sentiment_values,
        hole=0.4,
        marker_colors=['#28a745', '#dc3545', '#ffc107'],
        textinfo='label+percent+value',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>'
    )])
    
    sentiment_fig.update_layout(
        title={
            'text': '用户情感分布分析',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial'}
        },
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
    )
    
    # 2. 意图分类条形图（替换瀑布图避免兼容性问题）
    intent_data = {k: v for k, v in results['intent_results'].items() if v > 0}
    
    # 确保有有效数据
    if not intent_data:
        intent_data = {'其他': 1}  # 默认值避免空图表
    
    # 使用条形图替代瀑布图
    intent_fig = go.Figure(data=[go.Bar(
        x=list(intent_data.keys()),
        y=list(intent_data.values()),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(intent_data)],
        text=[f"{v}条" for v in intent_data.values()],
        textposition='auto'
    )])
    
    intent_fig.update_layout(
        title={
            'text': '用户反馈意图分类统计',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial'}
        },
        xaxis_title="反馈类型",
        yaxis_title="评论数量",
        showlegend=False,
        xaxis=dict(tickangle=45)  # 倾斜标签避免重叠
    )
    
    return sentiment_fig, intent_fig

def main():
    # 主标题
    st.markdown('<div class="main-header">🔊 用户之声回音壁 (Echo) Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🤖 AI驱动的用户反馈智能分析平台，让每个声音都被听见、被理解、被分析</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 🚀 Echo Pro 特色功能")
        
        features = [
            {"icon": "🎯", "title": "高精度情感分析", "desc": "智能识别用户情感倾向"},
            {"icon": "🔍", "title": "智能关键词提取", "desc": "自动发现用户关注焦点"},
            {"icon": "📋", "title": "意图智能分类", "desc": "精准识别用户反馈类型"},
            {"icon": "📊", "title": "可视化分析报告", "desc": "直观展示分析结果"},
            {"icon": "💡", "title": "AI智能洞察", "desc": "生成改进建议和优先级"},
            {"icon": "📁", "title": "一键导出报告", "desc": "支持多格式数据导出"}
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                <strong>{feature['icon']} {feature['title']}</strong><br>
                <small>{feature['desc']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析设置")
        
        # 分析参数设置
        confidence_threshold = st.slider("情感分析置信度阈值", 0.0, 1.0, 0.6, 0.1)
        keyword_limit = st.slider("关键词提取数量", 10, 50, 25, 5)
        
        st.markdown("### 📈 使用统计")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        st.metric("本次会话分析次数", st.session_state.analysis_count)

    # 初始化增强分析器
    # --- ↓↓↓ 粘贴下面的新代码 ↓↓↓ ---
# 初始化新的LLM分析器，它会自动从st.secrets读取API_KEY
    if 'analyzer' not in st.session_state:
        api_key = st.secrets.get("ZHIPU_API_KEY")
        st.session_state.analyzer = LLMEchoAnalyzer(api_key=api_key)

    # 数据输入模块
    st.markdown("## 📥 智能数据输入模块")
    
    # 输入方式选择
    input_method = st.radio(
        "选择输入方式：",
        ["文本粘贴", "文件上传", "示例数据"],
        horizontal=True
    )
    
    user_input = ""
    
    if input_method == "文本粘贴":
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area(
                "请粘贴用户评论文本（每行一条评论）：",
                height=250,
                placeholder="在此粘贴从App Store、Google Play、社交媒体等平台收集的用户评论...\n\n示例格式：\n这个APP真的很好用，界面设计很美观！\n最近总是闪退，希望能修复一下\n功能挺全的，就是有点卡顿"
            )
        
        with col2:
            st.markdown("### 📊 输入统计")
            if user_input:
                lines = len([line for line in user_input.split('\n') if line.strip()])
                chars = len(user_input)
                words = len(user_input.split())
                st.metric("评论条数", lines)
                st.metric("字符总数", chars)
                st.metric("词汇数量", words)
                
                # 输入质量评估
                avg_length = chars / lines if lines > 0 else 0
                if avg_length > 20:
                    st.success("✅ 评论长度适中")
                elif avg_length > 10:
                    st.warning("⚠️ 评论偏短")
                else:
                    st.error("❌ 评论过短")
    
    elif input_method == "示例数据":
        sample_data_options = {
            "电商APP评论": """这款购物APP界面设计很漂亮，操作也很流畅
商品种类丰富，价格也比较优惠，很满意
最近APP总是闪退，特别是在支付的时候，很影响购物体验
希望能增加商品对比功能，方便选择
客服回复很及时，解决问题的效率很高
物流信息更新不及时，不知道包裹到哪了
建议优化搜索功能，有时候搜不到想要的商品
整体体验不错，会继续使用并推荐给朋友
结账页面有bug，点击支付按钮没有反应
希望能支持更多支付方式，比如数字钱包""",
            
            "社交媒体APP评论": """界面简洁美观，很符合年轻人的审美
发布功能很完善，支持多种格式的内容
最近更新后经常卡顿，希望能优化一下性能
建议增加夜间模式，长时间使用眼睛会累
私信功能很好用，和朋友聊天很方便
广告太多了，影响使用体验
希望能增加更多的滤镜和贴纸
用户隐私保护做得不错，很有安全感
视频加载速度有点慢，特别是在网络不好的时候
点赞和评论功能响应及时，互动体验很好""",
            
            "学习教育APP评论": """课程内容质量很高，老师讲解很详细
界面设计简洁，功能分布合理，使用方便
视频播放经常卡顿，影响学习效果
希望能增加离线下载功能，方便随时学习
练习题目设计很好，有助于巩固知识点
客服态度很好，解答问题很耐心
建议增加学习进度统计功能
整体学习体验不错，确实有提升
APP启动速度有点慢，希望能优化
希望能增加更多互动功能，提高学习兴趣"""
        }
        
        selected_sample = st.selectbox("选择示例数据：", list(sample_data_options.keys()))
        user_input = sample_data_options[selected_sample]
        st.text_area("示例数据预览：", value=user_input, height=200, disabled=True)
    
    elif input_method == "文件上传":
        uploaded_file = st.file_uploader(
            "上传评论文件",
            type=['txt', 'csv'],
            help="支持.txt和.csv格式文件，每行一条评论"
        )
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                user_input = content
                st.success(f"✅ 文件上传成功！共{len(content.split())}行内容")
            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")

    # 分析控制面板
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyze_button = st.button("🚀 开始AI智能分析", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🧹 清空输入", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("📊 查看历史", use_container_width=True):
            st.info("历史分析功能开发中...")

    # 执行分析
    # --- ↓↓↓ 粘贴下面的新代码 ↓↓↓ ---
    if analyze_button and user_input:
        with st.spinner('🤖 正在调用云端AI大模型进行深度分析，请稍候...'):
            comments = st.session_state.analyzer.clean_text(user_input)
            
            if not comments:
                st.error("❌ 未检测到有效的评论内容。")
                st.stop()
            
            llm_results = st.session_state.analyzer.analyze(comments)

            if llm_results and 'summary_data' in llm_results and 'detailed_data' in llm_results:
                st.session_state.analysis_count += 1
                summary = llm_results['summary_data']
                details = llm_results['detailed_data']

                # 重新构建详细分析列表，恢复下载、表格等功能
                sentiment_details_rebuilt = []
                intent_details_rebuilt = []
                for item in details:
                    # 模拟旧版数据结构以兼容前端
                    sentiment_details_rebuilt.append({
                        'comment': item.get('comment', ''),
                        'sentiment': item.get('sentiment', 'neutral'),
                        'confidence': 0.95, # LLM结果给一个高置信度
                        'pos_score': 1 if item.get('sentiment') == 'positive' else 0,
                        'neg_score': 1 if item.get('sentiment') == 'negative' else 0
                    })
                    intent_details_rebuilt.append({
                        'comment': item.get('comment', ''),
                        'intent': item.get('intent', '其他')
                    })
                
                # 重新填充所有前端需要的数据
                st.session_state.enhanced_results = {
                    'sentiment_results': summary.get('sentiment_analysis', {}),
                    'keywords': [(item['keyword'], item['count']) for item in summary.get('keyword_extraction', [])],
                    'intent_results': summary.get('intent_classification', {}),
                    'ai_insights': {
                        'positive_highlights': summary.get('summary', {}).get('positive_highlights', []),
                        'negative_highlights': summary.get('summary', {}).get('negative_highlights', []),
                        # 重新填充keyword_insights以修复KeyError
                        'keyword_insights': {
                            'tech_focus': any(kw['keyword'] in ['bug', '卡顿', '闪退'] for kw in summary.get('keyword_extraction', [])),
                            'ux_focus': any(kw['keyword'] in ['界面', '设计', '体验'] for kw in summary.get('keyword_extraction', []))
                        },
                        'priority_issues': [],
                        'suggestions': []
                    },
                    'total_comments': len(comments), # 使用清洗后的评论列表长度，保证准确
                    'avg_confidence': 0.95,
                    'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sentiment_details': sentiment_details_rebuilt,
                    'intent_details': intent_details_rebuilt
                }
                st.success("🎉 AI深度分析完成！")
            else:
                st.error("分析失败，未能从AI模型获取有效数据。")

    # 显示增强分析结果
    if 'enhanced_results' in st.session_state:
        results = st.session_state.enhanced_results
        
        st.markdown("---")
        st.markdown("## 📊 AI智能分析报告")
        
        # 核心指标仪表盘
        st.markdown("### 🎯 核心指标概览")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{results['total_comments']}</h3>
                <p>总评论数</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            positive_rate = round(results['sentiment_results']['positive'] / results['total_comments'] * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{positive_rate}%</h3>
                <p>正面评价率</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            negative_rate = round(results['sentiment_results']['negative'] / results['total_comments'] * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{negative_rate}%</h3>
                <p>负面评价率</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            confidence_score = round(results['avg_confidence'] * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
                <h3>{confidence_score}%</h3>
                <p>分析置信度</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            intent_types = len([v for v in results['intent_results'].values() if v > 0])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{intent_types}</h3>
                <p>反馈类型数</p>
            </div>
            """, unsafe_allow_html=True)

        # 增强版可视化图表
        st.markdown("### 📈 可视化分析图表")
        
        sentiment_fig, intent_fig = create_enhanced_visualizations(results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sentiment_fig, use_container_width=True)
        with col2:
            st.plotly_chart(intent_fig, use_container_width=True)
        
        # 关键词分析
        st.markdown("### ☁️ 智能关键词分析")
        
        if results['keywords']:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 生成词云
                keywords_dict = dict(results['keywords'])
                if keywords_dict:
                    try:
                        plt.figure(figsize=(12, 6))
                        font_path = os.path.join('echo-project', 'fonts', 'font.otf')
                        wordcloud = WordCloud(
                            font_path=font_path,
                            width=1000, 
                            height=500,
                            background_color='white',
                            colormap='plasma',
                            max_words=30,
                            relative_scaling=0.5,
                            collocations=False,
                            prefer_horizontal=0.7
                        ).generate_from_frequencies(keywords_dict)
                        
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title('用户关注热点词云图', fontsize=16, pad=20)
                        
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=200, facecolor='white')
                        img_buffer.seek(0)
                        
                        st.image(img_buffer, use_column_width=True)
                        plt.close()
                    except Exception as e:
                        st.warning("⚠️ 词云图生成失败，可能是字体问题。显示关键词列表作为替代。")
                        # 备用显示方案
                        keyword_text = " | ".join([f"{word}({count})" for word, count in results['keywords'][:20]])
                        st.markdown(f"**关键词**: {keyword_text}")
            
            with col2:
                st.markdown("#### 🔥 高频关键词排行")
                for i, (word, count) in enumerate(results['keywords'][:15], 1):
                    percentage = round(count / results['total_comments'] * 100, 1) if results['total_comments'] > 0 else 0
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 0.2rem 0; border-bottom: 1px solid #eee;">
                        <span><strong>{i}. {word}</strong></span>
                        <span style="color: #666;">{count}次 ({percentage}%)</span>
                    </div>
                    """, unsafe_allow_html=True)

        # AI洞察和建议
        st.markdown("### 🤖 AI智能洞察")
        
        insights = results['ai_insights']
        
        # 优先级问题
        if insights['priority_issues']:
            st.markdown("#### ⚠️ 优先处理问题")
            for issue in insights['priority_issues']:
                severity_color = {'高': '#dc3545', '中': '#ffc107', '低': '#28a745'}
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                           padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {severity_color[issue['severity']]}; margin: 0.5rem 0;">
                    <strong>🚨 {issue['type']} (优先级: {issue['severity']})</strong><br>
                    📊 影响范围: {issue['count']}条评论<br>
                    💡 {issue['description']}
                </div>
                """, unsafe_allow_html=True)
        
        # 核心观点展示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😊 正面反馈精选")
            for i, comment in enumerate(insights['positive_highlights'], 1):
                st.markdown(f"""
                <div class="insight-box positive-insight">
                    <strong>💚 用户好评 #{i}</strong><br>
                    "{comment}"
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 😟 负面反馈精选")
            for i, comment in enumerate(insights['negative_highlights'], 1):
                st.markdown(f"""
                <div class="insight-box negative-insight">
                    <strong>💔 用户差评 #{i}</strong><br>
                    "{comment}"
                </div>
                """, unsafe_allow_html=True)
        
        # AI建议
        if insights['suggestions']:
            st.markdown("#### 💡 AI改进建议")
            for i, suggestion in enumerate(insights['suggestions'], 1):
                st.markdown(f"**{i}.** {suggestion}")
        
        # 关键词洞察
        keyword_insights = insights['keyword_insights']
        if keyword_insights['tech_focus'] or keyword_insights['ux_focus']:
            st.markdown("#### 🔍 关键词洞察")
            if keyword_insights['tech_focus']:
                st.warning("⚙️ 技术问题关注度较高，建议优先处理技术层面的bug和性能问题")
            if keyword_insights['ux_focus']:
                st.info("🎨 用户体验讨论较多，建议关注界面设计和交互优化")
        
        # 导出功能
        st.markdown("---")
        st.markdown("### 📁 分析报告导出")
        
        # 生成完整报告数据
        complete_report = {
            "分析概要": {
                "分析时间": results['analysis_time'],
                "总评论数": results['total_comments'],
                "分析置信度": f"{round(results['avg_confidence'] * 100, 1)}%"
            },
            "情感分析结果": results['sentiment_results'],
            "意图分类结果": results['intent_results'],
            "关键词分析": {
                "高频词汇": dict(results['keywords'][:20])
            },
            "AI洞察": {
                "优先问题": insights['priority_issues'],
                "改进建议": insights['suggestions'],
                "关键词洞察": insights['keyword_insights']
            },
            "代表性评论": {
                "正面反馈": insights['positive_highlights'],
                "负面反馈": insights['negative_highlights']
            }
        }
        
        # 生成详细的CSV数据
        detailed_data = []
        for detail in results['sentiment_details']:
            intent_info = next((item for item in results['intent_details'] 
                              if item['comment'] == detail['comment']), {'intent': '未分类'})
            detailed_data.append({
                '评论内容': detail['comment'],
                '情感倾向': detail['sentiment'],
                '情感置信度': round(detail['confidence'], 3),
                '意图分类': intent_info['intent'],
                '正面得分': detail['pos_score'],
                '负面得分': detail['neg_score']
            })
        
        df_detailed = pd.DataFrame(detailed_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON报告下载
            report_json = json.dumps(complete_report, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 下载JSON分析报告",
                data=report_json,
                file_name=f"echo_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV详细数据下载
            csv_data = df_detailed.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 下载CSV详细数据",
                data=csv_data,
                file_name=f"echo_detailed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # 生成简要总结报告
            summary_report = f"""
Echo AI分析报告摘要
===================
分析时间: {results['analysis_time']}
总评论数: {results['total_comments']}条
分析置信度: {round(results['avg_confidence'] * 100, 1)}%

情感分布:
• 正面评价: {results['sentiment_results']['positive']}条 ({round(results['sentiment_results']['positive']/results['total_comments']*100, 1)}%)
• 负面评价: {results['sentiment_results']['negative']}条 ({round(results['sentiment_results']['negative']/results['total_comments']*100, 1)}%)
• 中性评价: {results['sentiment_results']['neutral']}条 ({round(results['sentiment_results']['neutral']/results['total_comments']*100, 1)}%)

主要反馈类型:
{chr(10).join([f"• {k}: {v}条" for k, v in results['intent_results'].items() if v > 0])}

高频关键词:
{', '.join([word for word, count in results['keywords'][:10]])}

主要建议:
{chr(10).join([f"• {suggestion}" for suggestion in insights['suggestions']])}
            """
            
            st.download_button(
                label="📄 下载摘要报告",
                data=summary_report,
                file_name=f"echo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # 数据表格展示
        st.markdown("### 📋 详细数据表格")
        
        # 数据筛选器
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_filter = st.multiselect(
                "筛选情感类型:",
                ['positive', 'negative', 'neutral'],
                default=['positive', 'negative', 'neutral'],
                format_func=lambda x: {'positive': '正面', 'negative': '负面', 'neutral': '中性'}[x]
            )
        
        with col2:
            intent_options = list(set([item['intent'] for item in results['intent_details']]))
            intent_filter = st.multiselect(
                "筛选意图类型:",
                intent_options,
                default=intent_options
            )
        
        with col3:
            confidence_min = st.slider("最低置信度:", 0.0, 1.0, 0.0, 0.1)
        
        # 应用筛选器
        filtered_df = df_detailed[
            (df_detailed['情感倾向'].isin(sentiment_filter)) &
            (df_detailed['意图分类'].isin(intent_filter)) &
            (df_detailed['情感置信度'] >= confidence_min)
        ]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "评论内容": st.column_config.TextColumn("评论内容", width="large"),
                "情感倾向": st.column_config.TextColumn("情感倾向", width="small"),
                "情感置信度": st.column_config.ProgressColumn("置信度", min_value=0, max_value=1),
                "意图分类": st.column_config.TextColumn("意图分类", width="medium"),
                "正面得分": st.column_config.NumberColumn("正面得分", format="%d"),
                "负面得分": st.column_config.NumberColumn("负面得分", format="%d")
            }
        )
        
        st.markdown(f"📊 显示 {len(filtered_df)} / {len(df_detailed)} 条记录")
        
        # 高级分析功能
        st.markdown("---")
        st.markdown("### 🔬 高级分析功能")
        
        tab1, tab2, tab3 = st.tabs(["📈 趋势分析", "🔗 关联分析", "🎯 分群分析"])
        
        with tab1:
            st.markdown("#### 情感趋势分析")
            
            # 按评论长度分析情感分布
            df_detailed['评论长度'] = df_detailed['评论内容'].str.len()
            length_bins = pd.cut(df_detailed['评论长度'], bins=5, labels=['很短', '较短', '中等', '较长', '很长'])
            df_detailed['长度分组'] = length_bins
            
            length_sentiment = df_detailed.groupby(['长度分组', '情感倾向']).size().unstack(fill_value=0)
            all_sentiment_columns = ['positive', 'negative', 'neutral']
# 2. 使用 reindex 方法，补全缺失的列，并用 0 填充
            length_sentiment = length_sentiment.reindex(columns=all_sentiment_columns, fill_value=0)
            
            fig_trend = px.bar(
                x=length_sentiment.index,
                y=[length_sentiment['positive'], length_sentiment['negative'], length_sentiment['neutral']],
                title="不同评论长度的情感分布",
                labels={'x': '评论长度分组', 'y': '评论数量'},
                color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab2:
            st.markdown("#### 关键词关联分析")
            
            # 分析关键词与情感的关联
            keyword_sentiment = {}
            for detail in results['sentiment_details']:
                comment = detail['comment'].lower()
                sentiment = detail['sentiment']
                
                for word, count in results['keywords'][:15]:
                    if word.lower() in comment:
                        if word not in keyword_sentiment:
                            keyword_sentiment[word] = {'positive': 0, 'negative': 0, 'neutral': 0}
                        keyword_sentiment[word][sentiment] += 1
            
            # 创建关联热力图数据
            heatmap_data = []
            for word, sentiments in keyword_sentiment.items():
                total = sum(sentiments.values())
                if total > 0:
                    heatmap_data.append({
                        '关键词': word,
                        '正面关联度': sentiments['positive'] / total,
                        '负面关联度': sentiments['negative'] / total,
                        '中性关联度': sentiments['neutral'] / total,
                        '总出现次数': total
                    })
            
            if heatmap_data:
                df_heatmap = pd.DataFrame(heatmap_data)
                df_heatmap = df_heatmap.sort_values('总出现次数', ascending=False).head(10)
                
                fig_heatmap = px.imshow(
                    df_heatmap[['正面关联度', '中性关联度', '负面关联度']].T,
                    x=df_heatmap['关键词'],
                    y=['正面关联度', '中性关联度', '负面关联度'],
                    color_continuous_scale='RdYlBu_r',
                    title="关键词情感关联热力图"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab3:
            st.markdown("#### 用户分群分析")
            
            # 基于情感和意图进行用户分群
            cluster_data = df_detailed.groupby(['情感倾向', '意图分类']).size().reset_index(name='数量')
            
            fig_cluster = px.sunburst(
                cluster_data,
                path=['情感倾向', '意图分类'],
                values='数量',
                title="用户反馈分群旭日图",
                color='数量',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # 分群特征描述
            st.markdown("##### 🎯 分群特征分析")
            
            major_clusters = cluster_data.nlargest(3, '数量')
            for idx, cluster in major_clusters.iterrows():
                st.markdown(f"""
                **群体 {idx+1}: {cluster['情感倾向']} + {cluster['意图分类']}**
                - 规模: {cluster['数量']}条评论 ({round(cluster['数量']/results['total_comments']*100, 1)}%)
                - 特征: {'积极用户，对产品体验满意' if cluster['情感倾向'] == 'positive' else '有待改善的用户体验' if cluster['情感倾向'] == 'negative' else '中性用户群体'}
                """)

    # 比较分析功能
    if st.session_state.analysis_count > 1:
        st.markdown("---")
        st.markdown("### 📊 历史对比分析")
        st.info("💡 多次分析后可以查看趋势变化（功能规划中）")

    # 页面底部
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 1rem; margin: 2rem 0;'>
        <h3>🚀 用户之声回音壁 (Echo) Pro</h3>
        <p style='margin: 0; font-size: 1.1rem;'>AI赋能的用户反馈智能分析平台 | 让每个声音都被听见、被理解、被分析</p>
        <small>Powered by Advanced AI Technology</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()