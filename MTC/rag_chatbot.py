"""
RAG Чат-бот по конспектам
Запуск: python rag_chatbot.py
"""

import os
import sys
import subprocess
from typing import List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Функция для проверки и установки библиотек
def check_and_install_dependencies():
    """Проверяет наличие всех необходимых библиотек и устанавливает их при необходимости"""
    
    required_packages = [
        'langchain',
        'langchain-community',
        'chromadb',
        'pypdf',
        'sentence-transformers'
    ]
    
    missing_packages = []
    
    print("🔍 Проверка установленных библиотек...")
    
    for package in required_packages:
        package_import = package.replace('-', '_')
        try:
            __import__(package_import)
            print(f"   ✅ {package} установлен")
        except ImportError:
            print(f"   ❌ {package} не найден")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Устанавливаю отсутствующие библиотеки: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ {package} успешно установлен")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Ошибка при установке {package}: {e}")
                return False
    
    return True

# Проверяем зависимости перед импортом
if not check_and_install_dependencies():
    print("\n❌ Не удалось установить все зависимости. Попробуйте установить их вручную:")
    print("pip install langchain langchain-community chromadb pypdf sentence-transformers")
    input("\nНажмите Enter для выхода...")
    sys.exit(1)

# Теперь импортируем все необходимые библиотеки с правильными путями
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # В новых версиях LangChain Document находится в langchain_core
    try:
        from langchain_core.documents import Document
        print("✅ Импорт Document из langchain_core.documents")
    except ImportError:
        try:
            from langchain.schema import Document
            print("✅ Импорт Document из langchain.schema")
        except ImportError:
            # Создаем свой класс Document если ничего не работает
            class Document:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}
            print("✅ Используется встроенный класс Document")
    
    # В новых версиях text_splitter может быть в разных местах
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ Импорт text_splitter из langchain.text_splitter")
    except ImportError:
        try:
            from langchain_community.text_splitter import RecursiveCharacterTextSplitter
            print("✅ Импорт text_splitter из langchain_community.text_splitter")
        except ImportError:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                print("✅ Импорт text_splitter из langchain_text_splitters")
            except ImportError:
                # Свой простой сплиттер
                class RecursiveCharacterTextSplitter:
                    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None, length_function=len):
                        self.chunk_size = chunk_size
                        self.chunk_overlap = chunk_overlap
                        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                        self.length_function = length_function
                    
                    def split_documents(self, documents):
                        chunks = []
                        for doc in documents:
                            text = doc.page_content
                            # Простое разбиение по символам
                            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                                chunk_text = text[i:i + self.chunk_size]
                                if chunk_text:
                                    chunks.append(Document(
                                        page_content=chunk_text,
                                        metadata=doc.metadata
                                    ))
                        return chunks
                
                print("✅ Используется встроенный сплиттер")
    
    print("\n✅ Все библиотеки успешно загружены!")
    
except ImportError as e:
    print(f"\n❌ Ошибка импорта: {e}")
    print("\nПопробуйте выполнить команду:")
    print("pip install --upgrade langchain langchain-community langchain-core")
    input("\nНажмите Enter для выхода...")
    sys.exit(1)

@dataclass
class ChunkInfo:
    """Информация о чанке документа"""
    text: str
    page: int
    source: str
    relevance_score: float = 0.0

class SimpleRAGBot:
    """Простой RAG бот для работы с конспектами"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        print("\n🔄 Загрузка модели эмбеддингов...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ Модель эмбеддингов загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
        
        self.vector_store = None
        self.chunks_count = 0
        
        # Пробуем загрузить существующую БД
        self._load_existing_db()
    
    def _load_existing_db(self) -> bool:
        """Загрузка существующей базы данных"""
        if os.path.exists(self.persist_directory):
            try:
                print("🔄 Загрузка существующей базы данных...")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Получаем количество чанков
                self.chunks_count = len(self.vector_store.get()['ids'])
                print(f"✅ Загружена существующая БД с {self.chunks_count} фрагментами")
                return True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки БД: {e}")
                return False
        return False
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Обработка PDF файла"""
        if not os.path.exists(pdf_path):
            print(f"❌ Файл {pdf_path} не найден")
            return False
        
        print(f"\n📄 Загружаем PDF: {pdf_path}")
        
        try:
            # Загрузка PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"   ✅ Загружено {len(documents)} страниц")
            
            # Добавляем номера страниц
            for i, doc in enumerate(documents):
                doc.metadata["page"] = i + 1
                doc.metadata["source"] = os.path.basename(pdf_path)
            
            # Разбиение на чанки
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            print(f"   ✅ Создано {len(chunks)} фрагментов")
            
            # Создание векторного хранилища
            print("🔄 Создаем векторное представление...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vector_store.persist()
            self.chunks_count = len(chunks)
            
            print(f"✅ Готово! База данных сохранена в {self.persist_directory}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при обработке PDF: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[ChunkInfo]:
        """Поиск релевантных фрагментов"""
        if not self.vector_store:
            print("❌ Сначала загрузите PDF!")
            return []
        
        try:
            # Поиск с релевантностью
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            
            chunks = []
            for doc, score in results:
                chunks.append(ChunkInfo(
                    text=doc.page_content,
                    page=doc.metadata.get('page', 0),
                    source=doc.metadata.get('source', 'unknown'),
                    relevance_score=score
                ))
            
            return chunks
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            return []
    
    def generate_answer(self, question: str, chunks: List[ChunkInfo]) -> str:
        """Генерация ответа на основе найденных фрагментов"""
        if not chunks:
            return "❌ Не найдено информации по вашему вопросу."
        
        answer = []
        answer.append(f"\n{'='*60}")
        answer.append(f"📝 Вопрос: {question}")
        answer.append(f"{'='*60}\n")
        
        answer.append("🔍 Найдена следующая информация:\n")
        
        for i, chunk in enumerate(chunks, 1):
            answer.append(f"\n--- Источник {i} (Страница {chunk.page}) ---")
            answer.append(f"📊 Релевантность: {chunk.relevance_score:.3f}")
            # Обрезаем длинный текст для читаемости
            text_preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
            answer.append(f"📄 Текст: {text_preview}")
            answer.append("-" * 40)
        
        return "\n".join(answer)

def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    """Вывод меню"""
    print("\n" + "="*60)
    print("📚 RAG Чат-бот по конспектам")
    print("="*60)
    print("1. 📁 Загрузить/обработать PDF")
    print("2. ❓ Задать вопрос")
    print("3. 📊 Статистика")
    print("4. 🗑️ Очистить базу данных")
    print("5. 🚪 Выход")
    print("="*60)

def find_pdf_files():
    """Поиск PDF файлов в текущей директории"""
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    return pdf_files

def main():
    """Главная функция"""
    
    clear_screen()
    print("="*60)
    print("🚀 ЗАПУСК RAG ЧАТ-БОТА")
    print("="*60)
    
    try:
        bot = SimpleRAGBot()
    except Exception as e:
        print(f"\n❌ Ошибка при инициализации: {e}")
        input("\nНажмите Enter для выхода...")
        return
    
    while True:
        clear_screen()
        print_menu()
        
        # Показываем найденные PDF файлы
        pdf_files = find_pdf_files()
        if pdf_files:
            print("\n📁 Найденные PDF в текущей папке:")
            for i, pdf in enumerate(pdf_files, 1):
                size = os.path.getsize(pdf) / 1024  # размер в КБ
                print(f"   {i}. {pdf} ({size:.1f} KB)")
        else:
            print("\n📁 PDF файлы не найдены в текущей папке")
            print("   Положите PDF файл в эту папку и выберите пункт 1")
        
        print("\n" + "-"*60)
        choice = input("🔹 Выберите действие (1-5): ").strip()
        
        if choice == '1':
            clear_screen()
            print("📁 ЗАГРУЗКА PDF\n")
            
            # Если есть PDF файлы, предлагаем выбрать
            if pdf_files:
                print("Доступные PDF файлы:")
                for i, pdf in enumerate(pdf_files, 1):
                    size = os.path.getsize(pdf) / 1024
                    print(f"{i}. {pdf} ({size:.1f} KB)")
                print("0. Указать свой путь")
                
                file_choice = input("\nВыберите номер файла: ").strip()
                
                if file_choice.isdigit():
                    idx = int(file_choice)
                    if 1 <= idx <= len(pdf_files):
                        pdf_path = pdf_files[idx-1]
                        print(f"✅ Выбран файл: {pdf_path}")
                    else:
                        pdf_path = input("Введите полный путь к PDF файлу: ").strip()
                else:
                    pdf_path = input("Введите полный путь к PDF файлу: ").strip()
            else:
                pdf_path = input("Введите полный путь к PDF файлу: ").strip()
            
            if pdf_path and pdf_path.lower() != 'exit':
                bot.process_pdf(pdf_path)
            
            input("\nНажмите Enter для продолжения...")
        
        elif choice == '2':
            clear_screen()
            print("❓ ЗАДАТЬ ВОПРОС\n")
            
            if bot.chunks_count == 0:
                print("⚠️ Сначала загрузите PDF файл!")
                input("\nНажмите Enter для продолжения...")
                continue
            
            question = input("Ваш вопрос: ").strip()
            
            if question and question.lower() != 'exit':
                print("\n🔍 Ищем ответ...")
                chunks = bot.search(question)
                answer = bot.generate_answer(question, chunks)
                print(answer)
            
            input("\nНажмите Enter для продолжения...")
        
        elif choice == '3':
            clear_screen()
            print("📊 СТАТИСТИКА\n")
            print(f"📁 База данных: {bot.persist_directory}")
            print(f"📊 Фрагментов в БД: {bot.chunks_count}")
            print(f"🤖 Модель эмбеддингов: all-MiniLM-L6-v2")
            
            if bot.vector_store:
                print("✅ Статус: Активна")
            else:
                print("❌ Статус: Не активна (загрузите PDF)")
            
            # Показываем размер базы данных
            if os.path.exists(bot.persist_directory):
                import shutil
                size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                          for dirpath, _, filenames in os.walk(bot.persist_directory) 
                          for filename in filenames) / 1024 / 1024  # в МБ
                print(f"💾 Размер БД: {size:.2f} MB")
            
            input("\nНажмите Enter для продолжения...")
        
        elif choice == '4':
            clear_screen()
            print("🗑️ ОЧИСТКА БАЗЫ ДАННЫХ\n")
            
            confirm = input("Вы уверены? Все данные будут удалены! (да/нет): ").strip().lower()
            
            if confirm in ['да', 'yes', 'y', 'да']:
                import shutil
                if os.path.exists(bot.persist_directory):
                    shutil.rmtree(bot.persist_directory)
                    bot.vector_store = None
                    bot.chunks_count = 0
                    print("✅ База данных очищена")
                else:
                    print("❌ База данных не найдена")
            
            input("\nНажмите Enter для продолжения...")
        
        elif choice == '5':
            print("\n👋 До свидания!")
            break
        
        else:
            print("❌ Неверный выбор! Введите число от 1 до 5")
            input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        input("\nНажмите Enter для выхода...")