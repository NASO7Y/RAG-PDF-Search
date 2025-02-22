import streamlit as st
from arabic_support import support_arabic_text
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# إعداد واجهة التطبيق
st.set_page_config(page_title="البحث ضمن ملف PDF باستخدام Ollama و DeepSeek R1", layout="wide")
support_arabic_text(all=True)

st.title("البحث ضمن ملف PDF باستخدام Ollama و DeepSeek R1")

# تحميل ملف PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # حفظ الملف المرفوع في موقع مؤقت
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # تحميل ملف PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()  # إدراج الكتاب للقيام بالعمليات عليه

    # تقسيم النصوص إلى أجزاء
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # تهيئة نموذج التضمين لإنشاء متجهات للنصوص
    embedder = HuggingFaceEmbeddings()

    # إنشاء قاعدة بيانات البحث المتجهي باستخدام FAISS
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # تعريف نموذج اللغة وتحديد أنه سيستخدم نموذج DeepSeek R1 من خلال Ollama  
    llm = Ollama(model="deepseek-r1")  

    # إعداد التوجيهات للنموذج لتحديد كيفية استجابة الذكاء الاصطناعي
    prompt = """
    1. استخدم قطع السياق التالية للإجابة على السؤال في النهاية.
    2. إذا كنت لا تعرف الإجابة، قل فقط "لا أعرف" ولكن لا تختلق إجابة من تلقاء نفسك.
    3. اجعل الإجابة موجزة ومحدودة في 3-4 جمل.
    
    السياق: {context}  # يتم إدراج المعلومات المسترجعة هنا  
    السؤال: {question}  # يتم إدراج سؤال المستخدم هنا  
    إجابة مفيدة:
    """

    # إنشاء قالب التوجيهات وضبط التفاعل بين المدخلات والمخرجات
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # إعداد سلسلة المعالجة للإجابات
    llm_chain = LLMChain(
        llm=llm,  # استخدام نموذج DeepSeek R1
        prompt=QA_CHAIN_PROMPT,  # تمرير التوجيهات للنموذج
        callbacks=None,
        verbose=True
    )

    # دمج الوثائق المسترجعة مع مصدرها لعرضها داخل النظام
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="السياق:\nالمحتوى:{page_content}\nالمصدر:{source}",
    )

    # إعداد سلسلة استرجاع المعلومات وتوليد الإجابات
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,  # تمرير المعلومات المسترجعة إلى النموذج
        document_variable_name="context",  # تحديد المتغير الذي يحتوي على السياق
        document_prompt=document_prompt,
        callbacks=None
    )

    # ربط عملية الاسترجاع بالتوليد لضمان أن الإجابات تعتمد على المعلومات المسترجعة
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )

        # إدخال المستخدم للسؤال
    user_input = st.text_input("اطرح سؤالاً حول ملف PDF:")

        # معالجة إدخال المستخدم
    if user_input:
            with st.spinner("جاري المعالجة..."):
                response = qa(user_input)["result"]  # استرجاع البيانات ثم توليد الإجابة
                st.write("الإجابة:")
                st.write(response)  # عرض الإجابة النهائية

else:
    st.write("يرجى رفع ملف PDF للمتابعة.")
