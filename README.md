# RAG系統簡易實作

![示範](data/demo.gif)

## 概述

檢索增強生成（RAG）是一個強大的文件檢索、摘要和互動式問答工具。本專案利用 LangChain 和 Streamlit 提供一個無縫的網頁應用程式，讓使用者能夠執行這些任務。透過 RAG，你可以輕鬆上傳多個 PDF 文件，為文件內容生成向量嵌入，並與文件進行對話互動。聊天歷史記錄會被保留，以提供更好的互動體驗。

## 功能特色

- **Streamlit 網頁應用**：專案使用 Streamlit 建置，為使用者提供直觀且互動式的網頁界面。
- **輸入欄位**：使用者可以通過專用輸入欄位輸入 LLM 相關設定，支援 OpenAI API 金鑰或本地 LLM server URL。
- **文件上傳器**：使用者可以上傳多個 PDF 檔案，這些檔案隨後會被處理進行分析。
- **文件分割**：上傳的 PDF 會被分割成較小的文本塊，確保與具有 token 限制的模型相容。
- **向量嵌入**：文本塊會被轉換成向量嵌入，使檢索和問答任務更容易進行。
- **靈活的向量儲存**：使用 Chroma 作為本地向量存儲，無需額外的外部服務。
- **互動式對話**：使用者可以與文件進行互動式對話，提出問題並獲得答案。對話歷史會被保留以供參考。

## 系統需求

在運行專案之前，請確保你具備以下條件：

- Anaconda 或 Miniconda
- Python 3.7-3.9（建議使用 3.9）
- 以下選項其中之一：
  - OpenAI API 金鑰
  - 本地 LLM server（如 llama.cpp、text-generation-webui 等）
- 要上傳的 PDF 文件

## 使用方法

1. 下載並安裝 Anaconda：
   - 從 [Anaconda 官網](https://www.anaconda.com/download) 下載並安裝 Anaconda
   - 或者安裝更輕量的 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. 將儲存庫複製到本地電腦：
   ```bash
   git clone https://github.com/paul0728/RAG_system_simple_implementation.git
   cd RAG_LangChain_streamlit
   ```

3. 創建並啟動 Conda 環境：
   ```bash
   # 創建名為 rag 的環境，使用 Python 3.9
   conda create -n rag python=3.9
   
   # 啟動環境
   conda activate rag
   ```

4. 安裝所需的依賴套件：
   ```bash
   pip install -r requirements.txt
   ```

5. 運行 Streamlit 應用程式：
   ```bash
   streamlit run rag_engine.py
   ```

6. 打開網頁瀏覽器，訪問提供的 URL。

7. 設定 LLM：
   - 如果使用 OpenAI：輸入你的 API 金鑰
   - 如果使用本地 LLM：輸入你的 LLM server URL

8. 上傳你想要分析的 PDF 文件。

9. 點擊「提交文件」按鈕來處理文件並生成向量嵌入。

10. 在聊天輸入框中輸入你的問題，與文件進行互動式對話。

### 常見問題解決

如果你遇到以下錯誤：
```
DLL load failed while importing onnx_cpp2py_export: 動態連結程式庫 (DLL) 初始化例行程序失敗。
```

解決方案：
1. 確保使用 Python 3.7-3.9 版本的 Conda 環境
2. 在 Conda 環境中安裝 ONNX：
   ```bash
   conda install -c conda-forge onnx
   ```
3. 如果上述方法不起作用，可以嘗試：
   ```bash
   pip uninstall onnx
   pip install onnx==1.12.0
   ```

## 參考資料

- [RAG實作教學 | LangChain & LLaMA2 創造你的個人LLM](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4)
- [RAG實作教學 | Streamlit + LangChain + LLaMA2](https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-streamlit-langchain-llama2-c7d1dac2494e)