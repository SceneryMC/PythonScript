import os

from bs4 import BeautifulSoup


def extract_pdf_attachment_ids(html_content):
    """
    从 Zotero 生成的 HTML 报告中提取 PDF 附件的 ID。

    Args:
        html_content (str): HTML 报告的完整内容字符串。

    Returns:
        list: 包含所有 PDF 附件 ID 的字符串列表。
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # 最终存储所有 PDF ID 的列表
    pdf_ids = []

    # 1. 首先找到所有文献条目的 `li` 标签
    # 这些标签的 class 属性包含 'item'
    all_report_items = soup.find_all('li', class_='item')

    # 2. 遍历每一个文献条目
    for item in all_report_items:
        # 3. 在当前文献条目下，找到所有附件的 `li` 标签
        # 这些标签位于 class='attachments' 的 ul 标签内
        attachments_ul = item.find('ul', class_='attachments')

        # 检查是否存在附件列表
        if attachments_ul:
            attachment_lis = attachments_ul.find_all('li')

            # 4. 遍历该条目的所有附件
            for attachment_li in attachment_lis:
                # 获取附件 li 标签的文本内容，并去除首尾空白
                attachment_text = attachment_li.get_text(strip=True)

                # 5. 判断条件：
                # - 文本中包含 'PDF' (不区分大小写)
                # - 并且文本不是 'biburl' (也不区分大小写)
                if 'pdf' in attachment_text.lower() and 'biburl' not in attachment_text.lower():
                    # 如果满足条件，提取该 li 标签的 id
                    pdf_id = attachment_li.get('id')
                    if pdf_id:
                        pdf_ids.append(pdf_id)

    return pdf_ids


# --- 使用示例 ---

# 将你提供的 HTML 文本粘贴到这里
# 在实际使用中，你会从文件中读取它
html_text = """
<body>
    <ul class="report combineChildItems">
        <li id="item_4ICRKDBC" class="item conferencePaper">
        <h2>PANNS: enhancing graph-based approximate nearest neighbor search through recency-aware construction and parameterized search</h2>
            <table>
                <!-- ... 省略该条目的其他内容 ... -->
            </table>
            <h3 class="attachments">附件</h3>
            <ul class="attachments">
                <li id="item_9CJDVGRI">biburl					</li>
            </ul>
        </li>


        <li id="item_2XU3VYPJ" class="item journalArticle">
        <h2>Chameleon2++: An efficient chameleon2 clustering with approximate nearest neighbors</h2>
            <table>
                <!-- ... 省略该条目的其他内容 ... -->
            </table>
            <h3 class="attachments">附件</h3>
            <ul class="attachments">
                <li id="item_Q3HX85XW">biburl					</li>
                <li id="item_X4Q5RV9X">Preprint PDF					</li>
            </ul>
        </li>

        <li id="item_ABCDEFGH" class="item book">
        <h2>Another Book Title</h2>
            <table>
                <!-- ... 省略该条目的其他内容 ... -->
            </table>
            <h3 class="attachments">附件</h3>
            <ul class="attachments">
                <li id="item_12345678">Some other attachment</li>
                <li id="item_ZYXWVUT">Full Text PDF File</li>
            </ul>
        </li>

        <li id="item_NPDFJKLM" class="item report">
        <h2>A Report Without PDF</h2>
            <table>
                <!-- ... 省略该条目的其他内容 ... -->
            </table>
        </li>
    </ul>
</body>
"""

# 调用函数提取 ID
# found_ids = extract_pdf_attachment_ids(html_text)
#
# # 打印结果
# print("提取到的所有 PDF 附件 ID:")
# print(found_ids)

# --- 如何从文件中读取 HTML ---
# 在你的实际应用中，你会这样做：
try:
    with open(r'C:\Users\SceneryMC\Downloads\Zotero 报告.htm', 'r', encoding='utf-8') as f:
        html_from_file = f.read()
        ids_from_file = extract_pdf_attachment_ids(html_from_file)
        results = []
        for pdf_id in ids_from_file:
            path = rf'C:\Users\13308\Zotero\storage\{pdf_id.split('_')[1]}'
            pdfs = [x for x in os.listdir(path) if x.endswith('.pdf')]
            assert len(pdfs) == 1
            results.append(rf"{path}\{pdfs[0]}")
        print("从文件提取到的 ID:")
        print('\n'.join(results))
except FileNotFoundError:
    print("\n错误: 未找到 'zotero_report.html' 文件。")