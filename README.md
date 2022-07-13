# Topic Model

## **LDA模型**

LDA是 自然語言處理中相當有名的方法，是透過生成模型( 觀察大量資料，估計出資料的生成機制 )，在一系列文件中萃取出抽象的「主題」

LDA假設了 每篇文件都是由少數幾個「主題 (Topic)」所組成，而且每個主題都可以由少數幾個重要的「用詞 (Word)」描述。所以後續的實作中我們可以看到LDA會將每個文本分topic，且每個topic都有幾個重要的用詞。

![Untitled](Topic%20Model%2085344c5c42a846f8a179843930b25a80/Untitled.png)

本次實作除了實作LDA 模型訓練，並打包封裝成module

最後透過pyLDAvis做topic的視覺化呈現

![Untitled](Topic%20Model%2085344c5c42a846f8a179843930b25a80/Untitled%201.png)

## References

**[應用LDA建立topic model](https://wenwender.wordpress.com/2019/05/28/%e6%87%89%e7%94%a8lda%e5%bb%ba%e7%ab%8btopic-model/?preview_id=346&preview_nonce=1e925cf757&preview=true)**