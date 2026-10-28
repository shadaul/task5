import csv
import json
import os
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from numpy.polynomial import Polynomial
from scipy.stats import t


SHEET_URL = 'https://docs.google.com/spreadsheets/d/1WGGx7ewYQJmum-GjjLaCjVqUxCl5_f6VQ5ADF_XvZco/edit'
DATA_URL = os.environ.get('DATA_URL', SHEET_URL.replace('/edit', '/export?format=csv&gid=200225704'))
METHODS = ('IQR', 'Z-score', 'Moving average', 'Grubbs')
NAVY, TEAL, RED, GOLD = '#20364f', '#167d9a', '#cc3f45', '#c38613'
PAGE_SIZE = (8.27, 11.69)


@dataclass(frozen=True)
class Settings:
    iqr_factor: float = 1.5
    z_threshold: float = 3.0
    window: int = 7
    percent: float = 30.0
    alpha: float = 0.05

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Enter a number for each setting.')
        if self.iqr_factor < 0 or self.z_threshold <= 0 or self.percent < 0:
            raise ValueError('Check the detection thresholds.')
        if self.window < 1 or int(self.window) != self.window or not 0 < self.alpha < 1:
            raise ValueError('Check the window size and alpha.')

    def descriptions(self):
        return {
            'IQR': f'Q1 - {self.iqr_factor:g} x IQR; Q3 + {self.iqr_factor:g} x IQR',
            'Z-score': f'Absolute z-score > {self.z_threshold:g}',
            'Moving average': f'Previous {self.window} days; absolute deviation > {self.percent:g}%',
            'Grubbs': f'Two-sided, alpha = {self.alpha:g}; one extreme value',
        }


@dataclass
class Detection:
    values: pd.DataFrame
    note: str = ''


def prepare_data(frame):
    frame = frame.copy()
    frame.columns = [str(name).strip().lstrip('\ufeff') for name in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError('Two columns have the same mine name.')
    frame = frame.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
    unused = [c for c in frame if (not c or c.startswith('Unnamed:')) and frame[c].isna().all()]
    frame = frame.drop(columns=unused)
    if 'Date' not in frame or len(frame.columns) < 2 or frame.empty:
        raise ValueError('The sheet needs a Date column, a mine column and at least one row.')
    if any(not c or c.startswith('Unnamed:') for c in frame.columns):
        raise ValueError('Add a name to each mine column.')
    dates = frame['Date']
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = dates.astype('string').str.strip()
        parsed = pd.to_datetime(dates, format='%d.%m.%Y', errors='coerce')
        dates = parsed.fillna(pd.to_datetime(dates, format='%Y-%m-%d', errors='coerce'))
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Check for repeated or invalid dates. Use DD.MM.YYYY or YYYY-MM-DD.')
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Dates should not include a time.')
    frame['Date'] = dates
    for name in frame.columns.drop('Date'):
        column = frame[name]
        if not pd.api.types.is_numeric_dtype(column):
            column = column.astype('string').str.strip().str.replace(',', '.', regex=False)
        frame[name] = pd.to_numeric(column, errors='coerce').astype(float)
    outputs = frame.set_index('Date').sort_index()
    if not np.isfinite(outputs.to_numpy()).all():
        raise ValueError('Some output values are missing or invalid. Check the sheet.')
    if (outputs < 0).any().any():
        raise ValueError('Daily output cannot be negative.')
    if len(outputs) > 1 and not outputs.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError('Some dates are missing. Use 0 for days with no production.')
    if not np.isfinite(outputs.sum(axis=1).to_numpy()).all():
        raise ValueError('Output values are too large to add up.')
    return outputs


def parse_csv(text):
    header = next(csv.reader(StringIO(text.lstrip('\ufeff'))), [])
    names = [name.strip() for name in header]
    named = [name for name in names if name]
    if len(named) != len(set(named)):
        raise ValueError('The CSV has repeated column names.')
    return prepare_data(pd.read_csv(StringIO(text), dtype=str))


def load_data(url=DATA_URL):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https'):
        raise ValueError('DATA_URL must be an HTTP(S) CSV address.')
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != '_refresh']
    query.append(('_refresh', str(datetime.now(timezone.utc).timestamp())))
    fresh_url = urlunsplit(parts._replace(query=urlencode(query)))
    request = Request(fresh_url, headers={'Cache-Control': 'no-cache', 'User-Agent': 'MineAnalytics/1.0'})
    with urlopen(request, timeout=25) as response:
        text = response.read().decode('utf-8-sig')
    return parse_csv(text)


def add_total(outputs):
    total_name = 'Total'
    while total_name in outputs.columns:
        total_name += ' (all mines)'
    combined = outputs.copy()
    combined[total_name] = outputs.sum(axis=1)
    return combined, total_name


def statistics(combined):
    return pd.DataFrame({
        'Mean': combined.mean(), 'Std': combined.std(ddof=0),
        'Median': combined.median(),
        'IQR': combined.quantile(0.75) - combined.quantile(0.25),
    })


def detect(series, method, settings):
    mean, std = series.mean(), series.std(ddof=0)
    result = pd.DataFrame(index=series.index)
    result['Output'] = series
    result['Reference'] = mean
    result['Lower'] = np.nan
    result['Upper'] = np.nan
    result['Score'] = np.nan
    result['Limit'] = np.nan
    result['Assessed'] = True
    result['Flagged'] = False
    note = ''

    if method == 'IQR':
        q1, q3 = series.quantile([0.25, 0.75])
        result['Reference'] = series.median()
        result['Lower'] = q1 - settings.iqr_factor * (q3 - q1)
        result['Upper'] = q3 + settings.iqr_factor * (q3 - q1)
        result['Flagged'] = (series < result['Lower']) | (series > result['Upper'])
    elif method == 'Z-score':
        result['Lower'] = mean - settings.z_threshold * std
        result['Upper'] = mean + settings.z_threshold * std
        result['Score'] = (series - mean).abs() / std if std > 0 else 0.0
        result['Limit'] = settings.z_threshold
        result['Flagged'] = result['Score'] > settings.z_threshold
        if std == 0:
            note = 'All values are equal; no z-score outliers.'
    elif method == 'Moving average':
        average = series.shift(1).rolling(settings.window, min_periods=settings.window).mean()
        difference = (series - average).abs()
        margin = average.abs() * settings.percent / 100
        result['Reference'] = average
        result['Lower'], result['Upper'] = average - margin, average + margin
        score = difference.div(average.abs().replace(0, np.nan)) * 100
        score.loc[average.eq(0) & difference.eq(0)] = 0.0
        score.loc[average.eq(0) & difference.gt(0)] = np.inf
        result['Score'] = score
        result['Limit'] = settings.percent
        result['Assessed'] = average.notna()
        result['Flagged'] = result['Assessed'] & (difference > margin)
        note = f'First {min(settings.window, len(series))} days are skipped: not enough earlier data.'
    elif method == 'Grubbs':
        n, sample_std = len(series), series.std(ddof=1)
        result['Assessed'] = False
        note = 'Grubbs checks one extreme value and assumes roughly normal data.'
        if n >= 3 and sample_std > 0:
            critical_t = t.ppf(1 - settings.alpha / (2 * n), n - 2)
            critical_g = (n - 1) / np.sqrt(n) * np.sqrt(critical_t**2 / (n - 2 + critical_t**2))
            result['Lower'], result['Upper'] = mean - critical_g * sample_std, mean + critical_g * sample_std
            result['Score'] = (series - mean).abs() / sample_std
            result['Limit'] = critical_g
            candidate = result['Score'].idxmax()
            result.loc[candidate, 'Assessed'] = True
            result.loc[candidate, 'Flagged'] = result.loc[candidate, 'Score'] > critical_g
        else:
            note = 'Grubbs needs at least 3 values that are not all equal.'
    else:
        raise ValueError(f'Unknown method: {method}')
    result['Type'] = np.where(series > result['Reference'], 'Spike', 'Drop')
    return Detection(result, note)


def analyze(combined, settings):
    return {mine: {method: detect(combined[mine], method, settings) for method in METHODS}
            for mine in combined}


def anomaly_counts(results):
    return pd.DataFrame({mine: {name: int(test.values['Flagged'].sum())
                               for name, test in tests.items()}
                         for mine, tests in results.items()}).T


def trend_values(series, degree):
    if degree not in (1, 2, 3, 4):
        raise ValueError('Trend degree must be 1, 2, 3 or 4.')
    if len(series) <= degree:
        return None
    x = (series.index - series.index.min()).days.to_numpy()
    return Polynomial.fit(x, series.to_numpy(), degree)(x)


def draw_chart(ax, outputs, series, mask, chart_type, degree, total_name):
    if chart_type not in ('Line', 'Bar', 'Stacked'):
        raise ValueError('Unknown chart type.')
    colors = plt.get_cmap('tab20')
    if chart_type == 'Stacked' and series.name == total_name:
        bottom = np.zeros(len(outputs))
        for index, mine in enumerate(outputs):
            ax.bar(outputs.index, outputs[mine], bottom=bottom, width=0.8,
                   color=colors(index % 20), label=mine, linewidth=0)
            bottom += outputs[mine].to_numpy()
    elif chart_type in ('Bar', 'Stacked'):
        ax.bar(series.index, series, width=0.8, color=TEAL, label='Output')
    else:
        ax.plot(series.index, series, color=TEAL, linewidth=1.4,
                marker='o' if len(series) == 1 else None, label='Output')
    if mask.any():
        ax.scatter(series.index[mask], series[mask], color=RED, s=22,
                   edgecolors='white', linewidths=0.4, zorder=5, label='Anomalies')
    trend = trend_values(series, degree)
    if trend is not None:
        ax.plot(series.index, trend, color=GOLD, linestyle='--', linewidth=1.7,
                label=f'Trend (degree {degree})')
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.set_ylabel('Output (tonnes)', fontsize=9)
    ax.grid(axis='y', color='#dce3ea', alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.02)
    handles, labels = ax.get_legend_handles_labels()
    if chart_type == 'Stacked' and series.name == total_name and len(outputs.columns) > 6:
        handles, labels = handles[:2], labels[:2]
    ax.legend(handles, [textwrap.shorten(str(label), width=28, placeholder='...') for label in labels],
              fontsize=7, frameon=False, ncol=2)
    return trend


def report_signature(outputs, settings, chart_type, degree):
    digest = sha256(pd.util.hash_pandas_object(outputs, index=True).values.tobytes())
    digest.update(json.dumps([list(outputs.columns), asdict(settings), chart_type, degree],
                             ensure_ascii=False, sort_keys=True).encode())
    return digest.hexdigest()


def new_page(title, subtitle=''):
    page = plt.figure(figsize=PAGE_SIZE, facecolor='white')
    wrapped = textwrap.fill(str(title), width=54)
    title_size = min(17, 44 / (wrapped.count('\n') + 1))
    page.text(0.08, 0.915, wrapped, fontsize=title_size, weight='bold', va='top', color=NAVY)
    if subtitle:
        page.text(0.08, 0.85, subtitle, fontsize=9, color='#506176', va='top')
    return page


def table_pages(data, title, integer=False):
    rows, heights = [], []
    for name, row in data.iterrows():
        label = textwrap.fill(str(name), width=26, break_long_words=True)
        rows.append([label] + [str(int(value)) if integer else f'{value:,.2f}' for value in row])
        heights.append(max(0.038, 0.017 * (label.count('\n') + 1) + 0.014))
    start = 0
    while start < len(rows):
        end, height = start, 0.06
        while end < len(rows) and (height + heights[end] <= 0.72 or end == start):
            height += heights[end]
            end += 1
        page = new_page(title)
        ax = page.add_axes([0.08, 0.83 - height, 0.84, height])
        ax.axis('off')
        table = ax.table(cellText=rows[start:end],
                         colLabels=['Mine'] + [textwrap.fill(str(c), 13) for c in data.columns],
                         colWidths=[0.30] + [0.70 / len(data.columns)] * len(data.columns),
                         cellLoc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            cell.set_height((0.06 if r == 0 else heights[start + r - 1]) / height)
            cell.set_edgecolor('#d8e1eb')
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_facecolor(NAVY)
                cell.set_text_props(color='white', weight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f0f5f8')
        yield page
        start = end


def evidence(method, row):
    boundary = row['Upper'] if row['Type'] == 'Spike' else row['Lower']
    side = 'upper' if row['Type'] == 'Spike' else 'lower'
    text = f"{method}: {row['Type']} | {side} limit {boundary:.2f} t; difference {abs(row['Output'] - boundary):.2f} t."
    if method == 'IQR':
        text += f" Median {row['Reference']:.2f} t."
    else:
        label = {'Z-score': '|z|', 'Moving average': 'Deviation %', 'Grubbs': 'G'}[method]
        score = 'infinite (previous average is zero)' if np.isinf(row['Score']) else f"{row['Score']:.3f}"
        text += f" {label}: {score}; threshold {row['Limit']:.3f}."
    return text


def anomaly_pages(mine, series, tests):
    flags = pd.DataFrame({name: test.values['Flagged'] for name, test in tests.items()})
    dates = flags.index[flags.any(axis=1)]
    page, y = None, 0
    for date in dates:
        lines = []
        types = set()
        for method in METHODS:
            if flags.loc[date, method]:
                row = tests[method].values.loc[date]
                types.add(row['Type'])
                lines.extend(textwrap.wrap(evidence(method, row), width=86))
        height = 0.065 + 0.018 * len(lines)
        if page is None or y - height < 0.09:
            if page is not None:
                yield page
            page = new_page(f'Anomalies: {mine}', 'Dates, values and tests that detected them.')
            page.text(0.08, 0.055, 'Values are rounded for display.', fontsize=8, color='#506176')
            y = 0.79
        title = f"{date:%d %b %Y}  |  {' / '.join(sorted(types))}  |  {series.loc[date]:.2f} tonnes"
        page.text(0.08, y, title, fontsize=11, weight='bold', color=NAVY, va='top')
        page.text(0.08, y - 0.028, '\n'.join(lines), fontsize=9, va='top', linespacing=1.45)
        page.add_artist(Line2D([0.08, 0.92], [y - height + 0.012] * 2,
                               transform=page.transFigure, color='#d8e1eb', linewidth=0.6))
        y -= height
    if page is not None:
        yield page


def generate_report(outputs, combined, total_name, stats, results, settings,
                    chart_type, degree, loaded_at):
    buffer = BytesIO()
    counts = anomaly_counts(results)
    with PdfPages(buffer, metadata={'Title': 'Mine Analytics Report', 'Author': 'Mine Analytics'}) as pdf:
        number = 0

        def save(page):
            nonlocal number
            number += 1
            page.text(0.08, 0.025, 'Mine Analytics', fontsize=7, color='#66788a')
            page.text(0.92, 0.025, f'{number}', ha='right', fontsize=8, color='#66788a')
            try:
                pdf.savefig(page)
            finally:
                plt.close(page)

        page = new_page('Production report', f'{outputs.index.min():%d %b %Y} - {outputs.index.max():%d %b %Y}')
        text = [f'Mines: {len(outputs.columns)}     Days: {len(outputs)}',
                f'Data loaded: {loaded_at}', f'Charts: {chart_type}     Polynomial degree: {degree}', '',
                'Settings']
        for method, description in settings.descriptions().items():
            text.extend([f'{method}:', description, ''])
        text.extend(['Calculation notes',
                     'Each mine and the total are analysed separately.',
                     'Std and z-score use population standard deviation (ddof=0).',
                     'IQR = Q3 - Q1, using linear interpolation.',
                     'The moving average excludes the day being checked.',
                     'The first window is skipped. Positive output after a zero average',
                     'has infinite percentage deviation.',
                     'Grubbs tests one extreme value using sample standard deviation.',
                     'It assumes normal data and needs at least 3 nonconstant values.',
                     'A detected outlier does not necessarily mean an event occurred.',
                     'Charts mark outliers from all four tests.',
                     'Each date appears once, with the tests that detected it.',
                     'Stacked charts show each mine in the total. Single mines use bars.'])
        page.text(0.08, 0.79, '\n'.join(text), fontsize=9, va='top', linespacing=1.45)
        save(page)
        for page in table_pages(stats, 'Output statistics (tonnes)'):
            save(page)
        for page in table_pages(counts, 'Anomalies by method', integer=True):
            save(page)

        for mine in combined:
            series, tests = combined[mine], results[mine]
            union = pd.DataFrame({name: test.values['Flagged'] for name, test in tests.items()}).any(axis=1)
            page = new_page(f'Production: {mine}')
            values = stats.loc[mine]
            page.text(0.08, 0.835,
                      f"Mean {values['Mean']:.2f}   |   Std {values['Std']:.2f}   |   Median {values['Median']:.2f}   |   IQR {values['IQR']:.2f}",
                      fontsize=9, color=NAVY)
            ax = page.add_axes([0.12, 0.34, 0.80, 0.44])
            trend = draw_chart(ax, outputs, series, union, chart_type, degree, total_name)
            summary = [f'Dates with anomalies: {int(union.sum())}',
                       '  |  '.join(f'{name}: {counts.loc[mine, name]}' for name in METHODS)]
            if trend is None:
                summary.append(f'No trend: degree {degree} needs at least {degree + 1} days.')
            summary.extend(textwrap.wrap(tests['Moving average'].note, 90))
            summary.extend(textwrap.wrap(tests['Grubbs'].note, 90))
            page.text(0.08, 0.21, '\n'.join(summary), fontsize=9, va='top', linespacing=1.5)
            save(page)
            if chart_type == 'Stacked' and mine == total_name and len(outputs.columns) > 6:
                for start in range(0, len(outputs.columns), 24):
                    page = new_page('Mine colours')
                    for offset, name in enumerate(outputs.columns[start:start + 24]):
                        y = 0.81 - offset * 0.028
                        page.text(0.08, y, '\u25a0', color=plt.get_cmap('tab20')((start + offset) % 20), fontsize=12)
                        page.text(0.12, y, textwrap.shorten(str(name), 72, placeholder='...'), fontsize=9)
                    save(page)
            for page in anomaly_pages(mine, series, tests):
                save(page)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title='Mine Analytics', layout='wide')
    st.title('Mine analytics')
    st.write('Daily mine output')
    with st.sidebar:
        st.header('Source')
        st.markdown(f'[Open spreadsheet]({SHEET_URL})')
        refresh = st.button('Refresh data', key='refresh', width='stretch')
    if 'snapshot' not in st.session_state or refresh:
        try:
            fresh = load_data()
        except Exception as error:
            st.session_state['load_error'] = str(error)
        else:
            st.session_state['snapshot'] = fresh
            st.session_state['loaded_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            st.session_state.pop('load_error', None)
    if st.session_state.get('load_error'):
        st.error(f"Refresh failed: {st.session_state['load_error']}")
        if 'snapshot' not in st.session_state:
            st.info('Check that the sheet is shared and its CSV is available, then retry.')
            st.stop()
        st.warning('Still showing the previous data. Refresh again to load changes.')
    outputs = st.session_state['snapshot']
    loaded_at = st.session_state['loaded_at']
    combined, total_name = add_total(outputs)
    with st.sidebar:
        st.caption(f'Loaded {loaded_at}')
        st.header('Test settings')
        iqr = st.number_input('IQR factor', min_value=0.0, value=1.5, step=0.1, key='iqr')
        z = st.number_input('Z-score threshold', min_value=0.1, value=3.0, step=0.1, key='z')
        window = st.number_input('Moving average window (days)', min_value=1, value=7, step=1, key='window')
        percent = st.number_input('Moving average deviation (%)', min_value=0.0, value=30.0, step=5.0, key='percent')
        alpha = st.number_input('Grubbs alpha', min_value=0.001, max_value=0.1, value=0.05, step=0.001, format='%.3f', key='alpha')
        degree = st.selectbox('Trend degree', [1, 2, 3, 4], key='degree')
    settings = Settings(iqr, z, window, percent, alpha)
    stats = statistics(combined)
    results = analyze(combined, settings)
    counts = anomaly_counts(results)
    st.caption(f'{len(outputs.columns)} mines · {len(outputs)} days · {outputs.index.min():%d %b %Y} — {outputs.index.max():%d %b %Y}')
    if st.session_state.get('mine') not in combined.columns:
        st.session_state['mine'] = total_name
    columns = st.columns([2, 2, 1])
    mine = columns[0].selectbox('Mine or total', list(combined.columns), key='mine')
    method = columns[1].selectbox('Anomaly method', METHODS, key='method')
    chart_type = columns[2].selectbox('Chart type', ['Line', 'Bar', 'Stacked'], key='chart')
    for column, (label, value) in zip(st.columns(4), stats.loc[mine].items()):
        column.metric(label, f'{value:,.2f} t')
    if chart_type == 'Stacked' and mine != total_name:
        st.caption('Select Total to see all mines stacked together.')
    selected = results[mine][method]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    trend = draw_chart(ax, outputs, combined[mine], selected.values['Flagged'], chart_type, degree, total_name)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if trend is None:
        st.info(f'Trend degree {degree} requires at least {degree + 1} days of data.')
    st.caption(settings.descriptions()[method])
    if selected.note:
        st.caption(selected.note)
    overview, detail, report = st.tabs(['All mines', 'Anomalies', 'PDF report'])
    with overview:
        st.subheader('Output statistics')
        st.dataframe(stats.style.format('{:,.2f}'), width='stretch')
        st.subheader('Anomalies by test')
        st.dataframe(counts, width='stretch')
        st.caption('Tests may flag the same date, so do not add their counts. Std uses ddof=0.')
    with detail:
        flagged = selected.values.loc[selected.values['Flagged']].copy()
        st.subheader(f'{mine} · {method}')
        if flagged.empty:
            st.info('No anomalies found.')
        else:
            shown = flagged[['Output', 'Type', 'Reference', 'Lower', 'Upper']].copy()
            if method != 'IQR':
                shown['Test statistic'] = flagged['Score']
                shown['Threshold'] = flagged['Limit']
            shown.index = shown.index.strftime('%Y-%m-%d')
            shown.index.name = 'Date'
            st.dataframe(shown, width='stretch')
    with report:
        st.subheader('PDF report')
        st.write('Statistics, charts and anomalies for all mines and Total.')
        st.caption('The PDF includes all four tests. The chart above uses your selected test.')
        signature = report_signature(outputs, settings, chart_type, degree)
        if st.button('Create PDF', type='primary', key='generate'):
            with st.spinner('Creating PDF...'):
                try:
                    data = generate_report(outputs, combined, total_name, stats, results,
                                           settings, chart_type, degree, loaded_at)
                except Exception as error:
                    st.error(f'PDF failed: {error}')
                else:
                    st.session_state['report'] = (signature, data)
        saved = st.session_state.get('report')
        if saved and saved[0] == signature:
            st.download_button('Download PDF', data=saved[1], file_name='mine_report.pdf',
                               mime='application/pdf', on_click='ignore', key='download')
        elif saved:
            st.info('Data or settings changed. Create a new PDF.')


if __name__ == '__main__':
    main()


