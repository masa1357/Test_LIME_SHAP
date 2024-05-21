import pandas as pd
from pathlib import Path
from logging import getLogger, StreamHandler, Formatter
from sklearn.model_selection import train_test_split

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
    
    
def sprit_data_for_user(df, key, label, split_rate = [0.8, 0.2, 0], seed = 42):
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
        values="text",
        aggfunc=lambda x: " ".join(x),
    ).reset_index()
    df_pivoted.drop("new_column")

    # 結果を表示
    logger.debug(f"Pivot Info : {df_pivoted.shape}")
    logger.debug(f"Pivot columns : {df_pivoted.columns}")
    
    # trainの割合を計算
    train_df, tmp_df   = train_test_split(df, train_size=split_rate[0], random_state=seed, stratify=df[label], shuffle=True)
    valid_df, test_df  = train_test_split(tmp_df, test_size =split_rate[2], random_state=seed, stratify=df[label], shuffle=True)
    
    
    return train_df, valid_df, test_df


logger = set_logger()