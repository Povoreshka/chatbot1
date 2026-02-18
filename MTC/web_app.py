"""
Веб-версия RAG чат-бота с улучшенным дизайном
Запуск: python web_app.py
"""

import os
import sys
import subprocess
import tempfile
from typing import List
from dataclasses import dataclass
import warnings
import time
warnings.filterwarnings('ignore')

# Функция для установки streamlit если его нет
def install_streamlit():
    try:
        import streamlit as st
        print("✅ Streamlit уже установлен")
        return True, st
    except ImportError:
        print("📦 Устанавливаю Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit успешно установлен")
            import streamlit as st
            return True, st
        except:
            print("❌ Не удалось установить Streamlit")
            return False, None

# Проверяем и устанавливаем Streamlit
success, st = install_streamlit()
if not success:
    print("Попробуйте установить вручную: pip install streamlit")
    input("Нажмите Enter для выхода...")
    sys.exit(1)

# Импортируем наш RAG бот
try:
    from rag_chatbot import SimpleRAGBot, ChunkInfo
except ImportError:
    st.error("❌ Не найден файл rag_chatbot.py")
    st.stop()

# Настройка страницы
st.set_page_config(
    page_title="RAG Чат-бот по конспектам",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS для улучшения дизайна
st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .bot-message {
        background: #f8f9fa;
        border-left: 5px solid #667eea;
    }
    
    .source-box {
        background: #f1f3f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #20c997;
        margin: 0.5rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Анимации */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Поле ввода */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1.5rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Прогресс бар */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# Инициализация бота в сессии
@st.cache_resource
def init_bot():
    return SimpleRAGBot()

if 'bot' not in st.session_state:
    st.session_state.bot = init_bot()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Шапка
st.markdown("""
<div class="main-header fade-in">
    <h1>🎓 RAG Чат-бот по конспектам</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Загрузите PDF и задавайте вопросы по учебным материалам</p>
</div>
""", unsafe_allow_html=True)

# Создаем колонки для основного контента
col1, col2 = st.columns([2, 1])

with col1:
    # Заголовок чата
    st.markdown("### 💬 Чат с конспектами")
    
    # Контейнер для сообщений с прокруткой
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Задайте первый вопрос! Например: 'Что такое нейросети?'")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message fade-in">
                    <b>👤 Вы:</b><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message fade-in">
                    <b>🤖 Ассистент:</b><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Источники информации"):
                        for i, src in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <b>📄 Источник {i} (Страница {src.page})</b><br>
                                <small>Релевантность: {src.relevance_score:.2%}</small><br>
                                <p style="margin-top: 0.5rem;">{src.text[:200]}...</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Поле ввода вопроса
    st.markdown("---")
    col_input, col_button = st.columns([5, 1])
    
    with col_input:
        question = st.text_input(
            "Ваш вопрос:",
            placeholder="Например: Что такое нейронные сети?",
            label_visibility="collapsed",
            key="question_input"
        )
    
    with col_button:
        send_button = st.button("📤 Отправить", use_container_width=True)
    
    if send_button and question:
        if st.session_state.bot.chunks_count == 0:
            st.warning("⚠️ Сначала загрузите PDF файл в боковой панели!")
        else:
            # Добавляем вопрос пользователя
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Поиск и генерация ответа
            with st.spinner("🔍 Анализирую конспекты..."):
                # Прогресс бар для визуализации
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                chunks = st.session_state.bot.search(question)
                response = st.session_state.bot.generate_answer(question, chunks)
                
                # Добавляем ответ ассистента
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": chunks
                })
            
            st.rerun()

with col2:
    # Боковая панель с информацией
    st.markdown("### 📊 Информация")
    
    # Статус базы данных
    if st.session_state.bot.chunks_count > 0:
        st.markdown("""
        <div class="stat-card">
            <h4 style="color: #28a745; margin: 0;">✅ База знаний активна</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stat-card">
            <h4 style="color: #dc3545; margin: 0;">⏳ База знаний пуста</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Загрузка PDF
    st.markdown("### 📁 Загрузка материалов")
    
    uploaded_file = st.file_uploader(
        "Выберите PDF файл",
        type=['pdf'],
        help="Загрузите ваш конспект в формате PDF"
    )
    
    if uploaded_file is not None:
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        col_proc1, col_proc2 = st.columns(2)
        with col_proc1:
            if st.button("🔄 Обработать", type="primary", use_container_width=True):
                with st.spinner("Обработка документа..."):
                    success = st.session_state.bot.process_pdf(tmp_path)
                    if success:
                        st.success(f"✅ Готово! {st.session_state.bot.chunks_count} фрагментов")
                    else:
                        st.error("❌ Ошибка при обработке")
        
        with col_proc2:
            if st.button("🗑️ Отмена", use_container_width=True):
                st.rerun()
        
        # Удаляем временный файл
        os.unlink(tmp_path)
    
    st.markdown("---")
    
    # Статистика
    st.markdown("### 📈 Статистика")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <h3 style="color: #667eea; margin: 0;">{st.session_state.bot.chunks_count}</h3>
            <small>Фрагментов в БД</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="stat-card" style="text-align: center;">
            <h3 style="color: #667eea; margin: 0;">{len(st.session_state.messages) // 2}</h3>
            <small>Диалогов</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Размер БД
    if os.path.exists(st.session_state.bot.persist_directory):
        import shutil
        size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                  for dirpath, _, filenames in os.walk(st.session_state.bot.persist_directory) 
                  for filename in filenames) / 1024 / 1024
        st.markdown(f"""
        <div class="stat-card">
            <b>💾 Размер БД:</b> {size:.2f} MB
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Советы по использованию
    st.markdown("""
    ### 💡 Советы
    - Задавайте конкретные вопросы
    - Указывайте тему из конспекта
    - Проверяйте источники ответов
    - Используйте ключевые слова
    
    ### 🎯 Примеры вопросов
    - "Что такое градиентный спуск?"
    - "Объясни алгоритм обратного распространения"
    - "Какие бывают функции активации?"
    """)
    
    # Кнопка очистки истории
    if st.button("🗑️ Очистить историю чата", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Подвал
st.markdown("""
<div class="footer">
    🔍 <b>RAG (Retrieval-Augmented Generation)</b> — технология поиска ответов в ваших документах<br>
    <small>Сделано с ❤️ для студентов</small>
</div>
""", unsafe_allow_html=True)