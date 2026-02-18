import streamlit as st
import os
import tempfile
from rag_chatbot import SimpleRAGBot  # импортируем наш класс из консольной версии

# Настройка страницы
st.set_page_config(page_title="RAG Чат-бот", page_icon="📚")
st.title("📚 Чат-бот по конспектам")

# Инициализация бота
@st.cache_resource
def init_bot():
    return SimpleRAGBot()

bot = init_bot()

# Боковая панель для загрузки PDF
with st.sidebar:
    st.header("📁 Загрузка PDF")
    uploaded_file = st.file_uploader("Выберите PDF файл", type=['pdf'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if st.button("Обработать PDF"):
            with st.spinner("Обработка..."):
                bot.process_pdf(tmp_path)
                st.success("✅ PDF обработан!")
            os.unlink(tmp_path)

# Основной чат
st.header("💬 Задайте вопрос")

# Поле ввода вопроса
question = st.text_input("Ваш вопрос:", placeholder="Например: Что такое нейросети?")

if st.button("Спросить") and question:
    if bot.chunks_count == 0:
        st.warning("⚠️ Сначала загрузите PDF!")
    else:
        with st.spinner("🔍 Ищу ответ..."):
            chunks = bot.search(question)
            answer = bot.generate_answer(question, chunks)
            
            # Показываем ответ
            st.markdown("### Ответ:")
            st.write(answer)
            
            # Показываем источники
            if chunks:
                with st.expander("📖 Источники"):
                    for chunk in chunks:
                        st.markdown(f"""
                        **Страница {chunk.page}** (релевантность: {chunk.relevance_score:.3f})
                        > {chunk.text[:200]}...
                        ---
                        """)