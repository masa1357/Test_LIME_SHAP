import pandas as pd
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def set_logger():
    #? logger関連 ##########

    logger = getLogger(__name__)
    logger.setLevel('INFO')

    #? ハンドラが既に追加されているかをチェック
    if not logger.hasHandlers():
        #? 出力されるログの表示内容を定義
        formatter = Formatter(
            "%(asctime)s : %(name)s : %(levelname)s : %(lineno)s : %(message)s"
        )

        #? 標準出力のhandlerをセット
        stream_handler = StreamHandler()
        stream_handler.setLevel('INFO')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info('Test_message')
    
    return logger


def parser():
    arg_parser = ArgumentParser(add_help=True)

    ### Path settings
    arg_parser.add_argument('--reflection_root', type=str, required=True, help='path to Reflection dataset directory')
    arg_parser.add_argument('--grade_root', type=str, required=True, help='path to Grade dataset directory')
    
    arg_parser.add_argument('--key', type=str, required=True, help='Key to merging')
    arg_parser.add_argument('--split_rate', type=list, required=True, help='how rate to split')
    
    

def read_folder(path    :str, 
                rules   :str = "*.csv" ):
    """
    フォルダ内のファイルを全て読み込み，縦方向に結合する
    
    Parameters
    ----------
    path  : str
        フォルダパス
    rules : str
        読み込むフォルダの形式（デフォルトではcsvファイルすべて）
    
    Returns
    -------
    df : DataFrame
        読み込んだ全てのcsvを縦に結合したdf.
    """
    data_path = Path(path)
    logger.info(f"read {data_path} {rules} data...")
    
    file_list = list(data_path.glob(rules))
    logger.info(f"read : {file_list}")
    
    df = pd.DataFrame()
    if len(file_list) > 0:
        for i in file_list:
            temp_df = pd.read_csv(i)
            df = pd.concat([df, temp_df], axis=0, ignore_index=True)
        logger.info(f"get {len(df)} rows.")
    else:
        logger.error(f"No {rules} files in {data_path}")
        
    return df

def decode(df):
    # 'userid' と 'label' を固定して残りの列を縦に並べる
    df_melted = df.melt(
        id_vars=["userid", "grade"], var_name="text_label", value_name="text"
    )
    
    # 'text_label' から 'course_number' と 'question_number' を抽出
    df_melted[['course_number', 'question_number']] = df_melted['text_label'].str.split('-', expand=True)
    
    # 'text_label' 列を削除
    df_melted = df_melted.drop("text_label", axis=1)
    
    # 欠損値を削除
    df_melted = df_melted.dropna()

    return df_melted


    
def sprit_data_for_user(df, key, text, label, split_rate = [0.8, 0.2, 0], seed = 42):
    """
    データフレームを特定の列を基準に分割する．
    
    Parameters
    ----------
    df          : DataFrame
        対象df
    key         : str
        対象列名
    split_rate  : list(float)
        train, vaild, testの割合
    seed        : int
        seed値
    
    Returns
    -------
    train_df : DataFrame
    valid_df : DataFrame
    test_df  : DataFrame
        testはuse_Test = True としたときのみ出力
    """
    
    # 講義番号，質問番号ごとに列を作成
    def create_column_name(row):
        return f"{int(row['course_number']):02d}-{row['question_number']}"


    # 新しい列
    df["new_column"] = df.apply(create_column_name, axis=1)

    # 新しいデータフレームを定義
    df_pivoted = df.pivot_table(
        index=[key, label],
        columns="new_column",
        values=text,
        aggfunc=lambda x: " ".join(x),
    ).reset_index()
    logger.debug(f"columns: {df_pivoted.columns}")

    # 結果を表示
    logger.debug(f"Pivot Info : {df_pivoted.shape}")
    logger.debug(f"Pivot columns : {df_pivoted.columns}")
    
    
    # trainの割合を計算
    train_df, tmp_df   = train_test_split(df_pivoted, train_size=split_rate[0], random_state=seed, stratify=df_pivoted[label], shuffle=True)
    
    if split_rate[2] != 0:
        valid_size = split_rate[1] / (split_rate[1] + split_rate[2])
        valid_df, test_df  = train_test_split(tmp_df, test_size =1-valid_size, random_state=seed, stratify=tmp_df[label], shuffle=True)
        return decode(train_df), decode(valid_df), decode(test_df)
    else :
        return decode(train_df), decode(tmp_df)


def main():
    args = parser()
    
    left_df = read_folder(args.reflection_root)
    right_df= read_folder(args.grade_root)
    
    df = pd.merge(left=left_df, right=right_df, on=args.key)
    
    train = pd.DataFrame()
    valid = pd.DataFrame()
    test  = pd.DataFrame()
    
    train, valid, test = sprit_data_for_user(df, key=args.key, text="question_content", label="grade", split_rate=args.split_rate)
    train = decode(train)
    valid = decode(valid)
    test = decode(test)
    
    print(f"""
      columns = 
      train : {train.columns}
      valid : {valid.columns}
      test  : {test.columns}
      """
    )
    
    print(f"""
      rows = 
      train : {len(train)}\t- {len(train) / (len(train) + len(valid) + len(test))} %
      valid : {len(valid)}\t- {len(valid) / (len(train) + len(valid) + len(test))} %
      test  : {len(test)}\t- {len(test ) / (len(train) + len(valid) + len(test))} %
      """
    )
    
    return train, valid, test

if __name__ == '__main__':
    logger = set_logger()
    main()
    
    
    


logger = set_logger()