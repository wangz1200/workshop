from pathlib import Path
import pandas as pd
import iFinDPy as fdpy


class FutureData(object):

    def __init__(
            self,
            user: str = "zbhx003",
            password: str = "bb1705",
            with_login: bool = True,
    ):
        super().__init__()
        self.user = user
        self.password = password
        self.login_status = -1
        if with_login:
            self.login()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()

    def login(self):
        self.login_status = fdpy.THS_iFinDLogin(
            username=self.user,
            password=self.password,
        )
        if self.login_status != 0:
            raise Exception("Login failed")

    def logout(self):
        if self.login_status == 0:
            fdpy.THS_iFinDLogout()
            self.login_status = -1

    def fetch_data(
            self,
            code: str = "V2409.DCE",
            begin_time: str = "2024-05-23 09:00:00",
            end_time: str = "2024-05-23 23:00:00",
            fields: list | None = None,
            interval: int = 1,
            format_: str = "dataframe"
    ):
        fields = [
                     "open", "high", "low", "close", "volume", "amount", "openInterest",
                 ] or fields
        fields = ";".join(fields)
        rows = fdpy.THS_HF(
            thscode=code,
            jsonIndicator=fields,
            jsonparam='Fill:Original',
            begintime=begin_time,
            endtime=end_time,
            format=f"format:{format_}",
        )
        return rows

    def fetch_data_to_file(
            self,
            code: str,
            date_range: list,
            fields: list,
            saved_file_path: str,
            interval: int = 1,
            with_append: bool = False,
    ):
        fn_ = lambda x: [str(i) for i in x]
        ret = [
            ",".join(["time", "code", ] + fields) + "\n",
            ]
        for date in date_range:
            rows = fd.fetch_data(
                code=code,
                fields=fields,
                begin_time=date[0],
                end_time=date[1],
            )
            data = pd.DataFrame(rows.data)
            for idx in range(len(data)):
                row = fn_(data.loc[idx].to_list())
                ret.append(",".join(row) + "\n")
        if ret:
            mode = (
                "w+" if not with_append else "a+"
            )
            if with_append:
                ret = ret[1:]
            with Path(saved_file_path).open(
                    mode=mode, encoding="utf8"
            ) as f:
                f.writelines(ret)


if __name__ == "__main__":
    with FutureData() as fd:
        code = "V2409.DCE"
        fields = [
            "open", "high", "low", "close", "volume", "amount", "openInterest",
        ]
        date_range = [
            ["2023-09-01 09:00:00", "2023-09-30 23:00:00"],
            ["2023-10-01 09:00:00", "2023-10-31 23:00:00"],
            ["2023-11-01 09:00:00", "2023-11-30 23:00:00"],
            ["2023-12-01 09:00:00", "2023-12-31 23:00:00"],
            ["2024-01-01 09:00:00", "2024-01-31 23:00:00"],
            ["2024-02-01 09:00:00", "2024-02-29 23:00:00"],
            ["2024-03-01 09:00:00", "2024-03-31 23:00:00"],
            ["2024-04-01 09:00:00", "2024-04-30 23:00:00"],
            ["2024-05-24 09:00:00", "2024-05-31 23:00:00"],
        ]
        fd.fetch_data_to_file(
            code=code,
            date_range=date_range,
            fields=fields,
            saved_file_path=f"../data/future/{code}.csv",
            with_append=False,
        )
