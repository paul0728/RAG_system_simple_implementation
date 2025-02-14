# RAG系統簡易實作

![示範](data/demo.gif)

## 概述

檢索增強生成（RAG）是一個強大的文件檢索、摘要和互動式問答工具。本專案利用 LangChain、Streamlit 和 Pinecone 提供一個無縫的網頁應用程式，讓使用者能夠執行這些任務。透過 RAG，你可以輕鬆上傳多個 PDF 文件，為文件內容生成向量嵌入，並與文件進行對話互動。聊天歷史記錄會被保留，以提供更好的互動體驗。

## 功能特色

- **Streamlit 網頁應用**：專案使用 Streamlit 建置，為使用者提供直觀且互動式的網頁界面。
- **輸入欄位**：使用者可以通過專用輸入欄位輸入必要的憑證，如 LLM URL 或 OpenAI API 金鑰。
- **文件上傳器**：使用者可以上傳多個 PDF 檔案，這些檔案隨後會被處理進行分析。
- **文件分割**：上傳的 PDF 會被分割成較小的文本塊，確保與具有 token 限制的模型相容。
- **向量嵌入**：文本塊會被轉換成向量嵌入，使檢索和問答任務更容易進行。
- **靈活的向量儲存**：你可以選擇將向量嵌入儲存在本地向量存儲（Chroma）中。
- **互動式對話**：使用者可以與文件進行互動式對話，提出問題並獲得答案。對話歷史會被保留以供參考。

## 系統需求

在運行專案之前，請確保你具備以下條件：

- Python 3.7+
- LangChain
- Streamlit
- OpenAI API 金鑰
- 要上傳的 PDF 文件

## 使用方法

1. 將儲存庫複製到本地電腦：

   ```bash
   git clone https://github.com/paul0728/RAG_system_simple_implementation.git
   cd RAG_LangChain_streamlit
   ```

2. 建立並啟動虛擬環境：
   ```bash
   # 在 Windows 上
   python -m venv venv
   .\venv\Scripts\activate

   # 在 Linux 或 macOS 上
   python3 -m venv venv
   source venv/bin/activate
   ```

3. 安裝所需的依賴套件：
   ```bash
   pip install -r requirements.txt
   ```

4. 運行 Streamlit 應用程式：
   ```bash
   streamlit run rag_engine.py
   ```

5. 打開網頁瀏覽器，訪問提供的 URL。

6. 輸入你的 LLM URL 或 OpenAI API 金鑰。

7. 上傳你想要分析的 PDF 文件。

8. 點擊「提交文件」按鈕來處理文件並生成向量嵌入。

9. 在聊天輸入框中輸入你的問題，與文件進行互動式對話。

## 參考資料

- [RAG實作教學 | LangChain & LLaMA2 創造你的個人LLM](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4)
- [RAG實作教學 | Streamlit + LangChain + LLaMA2](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-streamlit-langchain-llama2-c7d1dac2494e)