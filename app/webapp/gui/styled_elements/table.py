from nicegui import ui
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Self

from nicegui import optional_features
from nicegui.element import Element
from nicegui.events import (
    GenericEventArguments,
    Handler,
    TableSelectionEventArguments,
    ValueChangeEventArguments,
    handle_event,
)
from webapp.config import ColorConfig

StickyOption = Literal['header', 'column', 'both', 'none']

class StickyTable(ui.table):
    _css_injected = False

    def __init__(self,
                 # height: Optional[str] = '300px',
                 # max_height: Optional[str] = None,
                 *args,
                 sticky: StickyOption = 'none',
                 theme: Optional[ColorConfig] = None,
                 **kwargs):
        self.theme = theme
        # Inject CSS once
        if not StickyTable._css_injected:
            ui.add_head_html('''
            <style>
            .sticky-header thead th {
                position: sticky;
                top: 0;
                background: #98194e;
                z-index: 1;
            }
            .sticky-column td:first-child,
            .sticky-column th:first-child {
                position: sticky;
                left: 0;
                z-index: 1;
            }
            .sticky-both thead th {
                position: sticky;
                top: 0;
                background: #98194e;
                z-index: 2;
            }
            .sticky-both td:first-child,
            .sticky-both th:first-child {
                position: sticky;
                left: 0;
                background: #98194e;
                z-index: 1;
            }
            </style>
            ''')
            StickyTable._css_injected = True

        # Determine class name based on sticky mode
        sticky_class = {
            'header': 'sticky-header',
            'column': 'sticky-column',
            'both': 'sticky-both',
            'none': '',
        }[sticky]

        # Call base constructor
        super().__init__(*args, **kwargs)
        self.classes(sticky_class)
        # if height:
        #     self.style(f'height: {height}; overflow: auto;')
        # elif max_height:
        #     self.style(f'max-height: {max_height}; overflow: auto;')


    @classmethod
    def from_pandas(cls,
                    df: 'pd.DataFrame', *,
                    sticky: Optional[StickyOption] = 'none',
                    theme: Optional[ColorConfig] = None,
                    columns: Optional[List[Dict]] = None,
                    column_defaults: Optional[Dict] = None,
                    row_key: str = 'id',
                    title: Optional[str] = None,
                    selection: Optional[Literal['single', 'multiple']] = None,
                    pagination: Optional[Union[int, dict]] = None,
                    on_select: Optional[Handler[TableSelectionEventArguments]] = None) -> Self:
        """Create a table from a Pandas DataFrame.

        Note:
        If the DataFrame contains non-serializable columns of type `datetime64[ns]`, `timedelta64[ns]`, `complex128` or `period[M]`,
        they will be converted to strings.
        To use a different conversion, convert the DataFrame manually before passing it to this method.
        See `issue 1698 <https://github.com/zauberzeug/nicegui/issues/1698>`_ for more information.

        *Added in version 2.0.0*

        :param theme:
        :param sticky:
        :param df: Pandas DataFrame
        :param columns: list of column objects (defaults to the columns of the dataframe)
        :param column_defaults: optional default column properties
        :param row_key: name of the column containing unique data identifying the row (default: "id")
        :param title: title of the table
        :param selection: selection type ("single" or "multiple"; default: `None`)
        :param pagination: a dictionary correlating to a pagination object or number of rows per page (`None` hides the pagination, 0 means "infinite"; default: `None`).
        :param on_select: callback which is invoked when the selection changes
        :return: table element
        """
        rows, columns_from_df = cls._pandas_df_to_rows_and_columns(df)
        table = cls(
            rows=rows,
            sticky=sticky,
            theme=theme,
            columns=columns or columns_from_df,
            column_defaults=column_defaults,
            row_key=row_key,
            title=title,
            selection=selection,
            pagination=pagination,
            on_select=on_select,
        )
        table._use_columns_from_df = columns is None
        return table
