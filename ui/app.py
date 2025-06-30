import argparse
from pathlib import Path
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import torch
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import base64, io
from pathlib import Path
import sys
from pathlib import Path
# --- Added for BPM calculation ---
from scipy.signal import find_peaks
# ensure project root on path so `src` can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vae_bilstm_attention import VAE
from src.visualization_vae import reconstruct_full_mean_std, ALPHA, ATTN_STRIDE, ATTN_WINDOW
# Parse command-line args for checkpoint
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='Path to model checkpoint')
args, unknown = parser.parse_known_args()
CKPT_PATH = Path(args.ckpt).resolve()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load model at startup
ckpt_data = torch.load(str(CKPT_PATH), map_location=DEVICE)
model = VAE(n_leads=12, n_latent=64).to(DEVICE)
if isinstance(ckpt_data, dict):
    model.load_state_dict(ckpt_data)
else:
    try:
        model.load_state_dict(ckpt_data.state_dict())
    except:
        model.load_state_dict(ckpt_data)
model.eval()

# Project root is two levels up from this file
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
MODELS_DIR = SRC_DIR / 'models'
DATA_DIR = ROOT / 'data'
IMG_DIR  = ROOT / 'img'


# Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'display': 'flex', 'height': '100vh'}, children=[
    # Sidebar
    html.Div(style={
            'width': '20%', 'padding': '20px', 'backgroundColor': '#f0f0f0'
        }, children=[
        html.H2("Configuration"),
        dcc.Upload(
            id='upload-data',
            children=html.Button(
                'Upload File (.npy, .dat, .hea)',
                style={'width': '100%', 'height': '50px', 'fontSize': '16px'}
            ),
            multiple=False,
            accept='.npy,.dat,.hea'
        ),
        html.Div(id='file-info', style={'marginTop': '10px'}),
        html.Br(),
        # Removed Model Checkpoint Path input block as per instructions
        html.Br(), html.Br(),
        html.Button(
            'Analyze ECG Sample',
            id='analyze-btn',
            n_clicks=0,
            style={'width': '100%', 'height': '50px', 'fontSize': '16px'}
        ),
        html.Br(), html.Br(),
        html.Label("Select Lead to Highlight:"),
        dcc.Dropdown(
            id='lead-select',
            options=[{'label': f'Lead {i+1:02d}', 'value': i} for i in range(12)],
            value=[],               # changed from None
            multi=True,             # added to allow multiple selections
            clearable=True,
            placeholder="Select one or more leads"
        ),
        html.Br(),
        html.Label("Anomaly Threshold:"),
        dcc.Slider(
            id='threshold-slider',
            min=0, max=1, step=0.01, value=0.5,
            marks={0: '0', 0.5: '0.5', 1: '1'}
        ),
        html.Br(),
        html.Label("Select Model:"),
        dcc.RadioItems(
            id='model-select',
            options=[
                {'label': 'VAE Model', 'value': 'vae'},
                {'label': 'CAE Model', 'value': 'cae'},
            ],
            value='vae',
            labelStyle={'display': 'block', 'marginBottom': '5px'}
        ),
    ]),
    # Main content
    html.Div(id='graph-container', style={'flex': '1', 'padding': '20px', 'overflowY': 'auto'})
])

@app.callback(
    Output('file-info', 'children'),
    Input('upload-data', 'filename')
)
def display_filename(fname):
    if fname:
        return html.Div(f"Selected file: {fname}")
    return html.Div("No file selected.")

def make_ecg_figure(x_orig_full, x_mean_full, x_std_full,
                    attn_full, mse_full, anomaly_full, filename,
                    lead_idxs=None, threshold=0.5, threshold_mask=None):
    T, n_leads = x_orig_full.shape
    if not lead_idxs:
        lead_idxs = list(range(n_leads))
    t = list(range(T))
    total_rows = n_leads * 4
    row_heights = []
    for _ in range(n_leads):
        # signal and anomaly equal height, then narrower attention and MSE
        row_heights += [0.4, 0.4, 0.1, 0.1]
    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.009,
        row_heights=row_heights,
        subplot_titles=sum([
            [f"Lead {lead_idxs[i]+1:02d}", "", "", ""]
            for i in range(n_leads)
        ], [])
    )
    for idx, lead in enumerate(lead_idxs):
        row_sig  = 4*idx + 1
        row_anom = 4*idx + 2
        row_attn = 4*idx + 3
        row_mse  = 4*idx + 4
        # original & reconstruction in the first row
        fig.add_trace(go.Scatter(
            x=t, y=x_orig_full[:,idx],
            # name=f"Orig L{lead+1:02d}",
            line=dict(color='black'),
            showlegend=False
        ), row=row_sig, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=x_mean_full[:,idx],
            # name=f"Recon L{lead+1:02d}",
            line=dict(dash='dash', color='red'),
            showlegend=False
        ), row=row_sig, col=1)
        # anomaly score line just below the signal
        fig.add_trace(go.Scatter(
            x=t, y=anomaly_full[:,idx],
            # name=f"Anomaly L{lead+1:02d}",
            line=dict(color='orange'),
            showlegend=False
        ), row=row_anom, col=1)
        # red overlay on segments where anomaly > threshold
        y_overlay = [val if val > threshold else None for val in anomaly_full[:,idx]]
        fig.add_trace(go.Scatter(
            x=t, y=y_overlay,
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False
        ), row=row_anom, col=1)
        # attention heatmap
        fig.add_trace(go.Heatmap(
            z=attn_full[:,idx][None,:], x=t,
            colorscale='Viridis',
            showscale=(idx==0),
            colorbar=dict(
                len=0.2,
                y=0.85 - idx*0.25,
                title="Attn",
                thickness=30,
                thicknessmode='pixels',
                x=1.02,
                xanchor='left',
                showticklabels=False
            )
        ), row=row_attn, col=1)
        # MSE heatmap
        fig.add_trace(go.Heatmap(
            z=mse_full[:,idx][None,:], x=t,
            colorscale='Blues',
            showscale=(idx==0),
            colorbar=dict(
                len=0.2,
                y=0.6 - idx*0.25,
                title="MSE",
                thickness=30,
                thicknessmode='pixels',
                x=1.05,
                xanchor='left'
            )
        ), row=row_mse, col=1)
    # add legend entries after all lead loops
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='black', width=2),
                             name='Original',
                             showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='red', dash='dash', width=2),
                             name='Reconstructed',
                             showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='orange', width=2),
                             name='Anomaly Score',
                             showlegend=True), row=1, col=1)
    # Hide y-axis for all but signal rows
    for idx in range(n_leads):
        row_sig = 4*idx + 1
        for row in [4*idx+2, 4*idx+3, 4*idx+4]:
            fig.update_yaxes(showticklabels=False, row=row, col=1)
    fig.update_layout(
        height=200*n_leads,
        title_text=f"ECG {filename} — 12-Lead Anomaly",
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.10,
            yanchor='bottom'
        ),
        margin=dict(t=80, b=20, l=50, r=200)
    )
    return fig

@app.callback(
    Output('graph-container', 'children'),
    Input('analyze-btn', 'n_clicks'),
    Input('threshold-slider', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('lead-select', 'value')
)
def analyze_ecg(n_clicks, threshold, contents, filename, lead_idx):
    if n_clicks == 0 or contents is None:
        return html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    sig = np.load(io.BytesIO(decoded))
    if sig.ndim==2 and sig.shape[0]!=12 and sig.shape[1]==12:
        sig = sig.T
    sig = torch.tensor(sig, dtype=torch.float32)

    # reconstruct
    recon_m, recon_s = reconstruct_full_mean_std(model, sig)
    # compute strips
    # attention, mse, anomaly per-lead
    from numpy import log, percentile
    # attention per-lead (reuse compute_strips logic inline)
    T = sig.shape[1]; n_leads=12
    attn_full = np.zeros((T,n_leads))
    for lead in range(n_leads):
        sig1 = torch.zeros_like(sig); sig1[lead]=sig[lead]
        wins=[]
        for start in range(0, T-ATTN_WINDOW+1, ATTN_STRIDE):
            wins.append(sig1[:,start:start+ATTN_WINDOW].T)
        win = torch.stack(wins).to(DEVICE)
        with torch.no_grad():
            *_, att = model.forward(win)
        a = att.mean(dim=(1,2,3) if att.ndim==4 else (1,2)).cpu().numpy()
        tmp=np.zeros(T); cnt=np.zeros(T)
        for i_w, s in enumerate(range(0, T-ATTN_WINDOW+1, ATTN_STRIDE)):
            tmp[s:s+ATTN_WINDOW]+=a[i_w]; cnt[s:s+ATTN_WINDOW]+=1
        cnt[cnt==0]=1; attn_full[:,lead]=tmp/cnt

    # mse & anomaly
    x_orig = sig.numpy().T; x_mean=recon_m.T.numpy(); x_std=recon_s.T.numpy()
    mse_full = (x_orig-x_mean)**2
    anomaly_full = ALPHA*mse_full + (1-ALPHA)*log(x_std**2+1e-6)

    threshold_mask = (anomaly_full > threshold)

    # --- Compute BPM from lead II (index 1) ---
    # compute BPM using lead II (index 1)
    ecg_ch = sig[1].cpu().numpy()
    peaks, _ = find_peaks(ecg_ch, distance=0.4 * 500)  # 0.4s min distance at 500Hz
    if len(peaks) >= 2:
        times = peaks / 500.0  # convert to seconds
        rr_intervals = np.diff(times)
        bpm_inst = 60.0 / rr_intervals
        bpm_mean = float(np.mean(bpm_inst))
    else:
        bpm_mean = float('nan')

    if lead_idx:
        # Subset arrays to only the selected leads
        x_orig = x_orig[:, lead_idx]
        x_mean = x_mean[:, lead_idx]
        x_std = x_std[:, lead_idx]
        attn_full = attn_full[:, lead_idx]
        mse_full = mse_full[:, lead_idx]
        anomaly_full = anomaly_full[:, lead_idx]
        threshold_mask = threshold_mask[:, lead_idx]

    fig = make_ecg_figure(x_orig, x_mean, x_std,
                          attn_full, mse_full, anomaly_full, filename,
                          lead_idxs=lead_idx,
                          threshold=threshold, threshold_mask=threshold_mask)
    # annotate title with BPM
    fig.update_layout(
        title_text=f"ECG {filename} — 12-Lead Anomaly (BPM={bpm_mean:.1f})"
    )

    # determine overall anomaly status
    is_anom = bool(np.any(threshold_mask))
    # display a single-sentence status
    if is_anom:
        status_div = html.P("This ECG shows anomalies.", style={'color': 'red', 'textAlign': 'center', 'fontWeight': 'bold'})
    else:
        status_div = html.P("This ECG appears normal.", style={'color': 'green', 'textAlign': 'center', 'fontWeight': 'bold'})

    # Disable line simplification for all scatter traces for full fidelity
    for trace in fig.data:
        if trace.type in ('scatter', 'scattergl'):
            trace.update(line_simplify=False)

    graph = dcc.Graph(
        figure=fig,
        style={'height': f'{90}vh'}  # use viewport height to give plenty of vertical space
    )
    return [status_div, graph]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
