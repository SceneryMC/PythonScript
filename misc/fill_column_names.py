import csv
import os.path

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison, TokenList
from sqlparse.tokens import Keyword, Name

from artist_follower.netease_artist_follower import headers


def extract_table_aliases(parsed):
    """提取 SQL 语句中的表及其别名"""
    aliases = {}  # {别名: 表名}
    from_seen = False

    for token in parsed.tokens:
        if token.is_keyword and token.value.upper() == "FROM":
            from_seen = True
            continue

        if from_seen:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    name = identifier.get_real_name()
                    alias = identifier.get_alias() or name
                    aliases[alias] = name
            elif isinstance(token, Identifier):
                name = token.get_real_name()
                alias = token.get_alias() or name
                aliases[alias] = name

            # `FROM` 之后的 `WHERE` 或 `ORDER BY` 代表表名解析结束
            if token.is_keyword and token.value.upper() in {"WHERE", "ORDER"}:
                break

    return aliases


def extract_columns_from_where(where_clause, table_aliases, table_aliases_reverse):
    """递归解析 WHERE 子句，提取涉及的列"""
    columns = set()

    if isinstance(where_clause, TokenList):
        def generate_output(t, c):
            return f"{table_aliases_reverse[t]}.{c}"
        
        for token in where_clause.tokens:
            if isinstance(token, Comparison):
                # 提取 't.id=mc.movie_id' 形式的左右操作数
                left, right = token.left, token.right
                if isinstance(left, Identifier):
                    col_name = left.get_real_name()
                    table_name = table_aliases.get(left.get_parent_name(), None)
                    if table_name:
                        columns.add(generate_output(table_name, col_name))
                if isinstance(right, Identifier):
                    col_name = right.get_real_name()
                    table_name = table_aliases.get(right.get_parent_name(), None)
                    if table_name:
                        columns.add(generate_output(table_name, col_name))
            elif isinstance(token, Identifier):
                # 直接的列名
                col_name = token.get_real_name()
                table_name = table_aliases.get(token.get_parent_name(), None)
                if table_name:
                    columns.add(generate_output(table_name, col_name))
            elif isinstance(token, TokenList):
                # 递归处理嵌套结构
                columns.update(extract_columns_from_where(token, table_aliases))

    return columns


def extract_columns(parsed, table_aliases, table_aliases_reverse):
    """ 提取 SQL 语句中 WHERE 子句涉及的所有列 """
    columns = set()

    for token in parsed.tokens:
        if isinstance(token, Where):
            columns.update(extract_columns_from_where(token, table_aliases, table_aliases_reverse))

    return columns


def rewrite_sql(sql):
    """解析 SQL 并将 COUNT(*) 替换为涉及的所有列"""
    parsed = sqlparse.parse(sql)[0]
    table_aliases = extract_table_aliases(parsed)  # 获取别名
    table_aliases_reverse = {id_: key for key, id_ in table_aliases.items()}
    columns = extract_columns(parsed, table_aliases, table_aliases_reverse)  # 使用别名解析 WHERE 条件

    if not columns:
        return sql  # 没有提取到列，保持原 SQL

    columns_str = ", ".join(sorted(columns))  # 按字母顺序排序，便于比较
    new_sql = sql.replace("COUNT(*)", columns_str)

    return new_sql


def translate_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(spamreader):
            replaced_sql = rewrite_sql(row[0].strip())
            with open(f'job_extra_5/{i}.sql', 'w') as f:
                f.write(replaced_sql)


if __name__ == '__main__':
    translate_csv('./job_light_ranges_subqueries.csv')
# 测试 SQL
# sql = "SELECT COUNT(*) FROM title t, movie_companies mc WHERE t.id=mc.movie_id AND mc.company_type_id=2;"
# new_sql = rewrite_sql(sql)
# print(new_sql)
