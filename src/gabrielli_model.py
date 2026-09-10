"""
Gabrielli (2021) subnet neural network for individual RBNS claims reserving.

PyTorch re-implementation adapted for MTPL data (10 development years).

The architecture builds one subnet per development year t=1..N_DY.  Each
subnet predicts:
  - P(payment > 0 at DY t)      (binary, via sigmoid)
  - E[log(payment) | payment>0]  (continuous, linear)

Key features:
  - Dual embeddings (input path + skip-connection/output path)
  - Past payment info as embedded categorical inputs with cascaded dropout
  - AY as an additive calendar-year effect
  - Shared hidden-layer architecture across subnets
"""

import math
import numpy as np
import torch
import torch.nn as nn

from src.gabrielli_data_prep import N_DY


# -----------------------------------------------------------------------
# Embedding dimensions for MTPL data (analogous to Gabrielli's constants)
# -----------------------------------------------------------------------
EMBED_DIMS = {
    "claim_type": 6,  # 6 MTPL claim types
    "AY": 10,  # 10 accident years
    "AQ": 4,  # binned notidel (4 buckets: <0.5yr, 0.5-1yr, 1-2yr, 2+yr)
    "report_dy": 6,  # (A) full range: RepDel 0-5 (uncapped)
    "pay_info": 6,  # 6 payment magnitude buckets
}

# Number of continuous numeric features concatenated alongside embeddings
N_NUMERIC_FEATURES = 3  # (B) notidel_raw, (D) log_cum_paid_obs, no_payments_raw

# Number of claim-level embedded features that feed into the feature path
N_CLAIM_EMBED_FEATURES = (
    3  # claim_type, AQ(binned notidel), report_dy (AY added separately)
)

# Total claim features = embedded + numeric
N_CLAIM_FEATURES = N_CLAIM_EMBED_FEATURES + N_NUMERIC_FEATURES  # 6

# Hidden layer sizes (same as Gabrielli: 40, 30, 10)
NEURONS = (40, 30, 10)


class GabrielliSubnet(nn.Module):
    """
    A single subnet for development year t.

    Predicts (indicator, mean) for one DY, given claim features and
    past payment information through DYs 1..(t-1).
    """

    def __init__(self, t, neurons=NEURONS):
        """
        Parameters
        ----------
        t : int
            Development year (1-based). Subnet for DY t uses past pay
            info from DY 1..(t-1).
        neurons : tuple
            (hidden1, hidden2, hidden3) layer sizes.
        """
        super().__init__()
        self.t = t
        self.n1, self.n2, self.n3 = neurons
        self.n_past_pay = t - 1  # number of past payment info inputs (0 for t=1)

        # Feature input -> hidden: features_input (N_CLAIM_FEATURES scalars)
        #   + Pay01Info..Pay{t-1}Info embeddings (t-1 scalars)
        input_size = (
            N_CLAIM_FEATURES + 1 + self.n_past_pay
        )  # +1 for Pay00 (always known, DY1 payment)
        # Actually: for t=1, input is just features; for t>1, features + Pay01..Pay{t-1}
        # DY 1 payment (Pay01Info) is always known (Time_Known01 is always 1 for reported claims).
        # Re-thinking: Gabrielli's Pay00 is always known. Our Pay01 is DY 1, which IS known for
        # all RBNS claims that have been reported (report_dy=1 means AY+1-1<=10 always).
        # Actually for AY=10, Time_Known01 = (10+1-1<=10) = True. So yes, DY1 always known.
        # For DY t, the past payments are Pay01Info..Pay{t-1}Info.
        # The FIRST past payment (Pay01Info) is always included without dropout (like Gabrielli's Pay00).
        # Subsequent past payments get dropout.

        # For t=1: no past payments at all.  input_size = N_CLAIM_FEATURES
        if t == 1:
            feat_in = N_CLAIM_FEATURES
        else:
            feat_in = N_CLAIM_FEATURES + (t - 1)  # features + past pay embeds

        self.fc_feat_to_hidden = nn.Linear(feat_in, self.n1)
        # NOTE: fc_ay_to_hidden is set externally by GabrielliNN to a SHARED
        # layer across all subnets, matching the R code and paper Section 4.2.4.

        # tanh activation (approximately identity near zero; Gabrielli Section 4.2.4)
        self.fc_hidden2 = nn.Linear(self.n1, self.n2)

        # Indicator head
        self.fc_ind_hidden = nn.Linear(self.n2, self.n3)
        # Skip connection input: N_CLAIM_FEATURES output embeds + past pay output embeds
        if t == 1:
            skip_in = N_CLAIM_FEATURES
        else:
            skip_in = N_CLAIM_FEATURES + (t - 1)
        self.fc_ind_out = nn.Linear(self.n3 + skip_in, 1)

        # Mean head
        self.fc_mean_hidden = nn.Linear(self.n2, self.n3)
        self.fc_mean_out = nn.Linear(self.n3 + skip_in, 1)

    def forward(
        self,
        features_in,
        ay_embed_in,
        past_pay_in,
        features_out,
        past_pay_out,
        ay_ind_effect,
        ay_mean_effect,
    ):
        """
        Parameters
        ----------
        features_in : (B, N_CLAIM_FEATURES)  input-path claim feature embeds
        ay_embed_in : (B, 1)                 input-path AY embed
        past_pay_in : (B, t-1) or None       input-path past payment embeds (dropout-masked)
        features_out : (B, N_CLAIM_FEATURES) output-path claim feature embeds
        past_pay_out : (B, t-1) or None      output-path past payment embeds (dropout-masked)
        ay_ind_effect : (B, 1)               AY indicator additive effect
        ay_mean_effect : (B, 1)              AY mean additive effect

        Returns
        -------
        indicator : (B, 1)  payment probability (sigmoid)
        mean : (B, 1)       log-payment mean (linear)
        """
        # Concatenate features + past pay for input path
        if past_pay_in is not None:
            x_in = torch.cat([features_in, past_pay_in], dim=1)
        else:
            x_in = features_in

        # Hidden layers
        h = self.fc_feat_to_hidden(x_in)
        h = h + self.fc_ay_to_hidden(ay_embed_in)
        h = torch.tanh(h)
        h = torch.tanh(self.fc_hidden2(h))

        # Indicator head
        h_ind = torch.tanh(self.fc_ind_hidden(h))
        if past_pay_out is not None:
            skip = torch.cat([h_ind, features_out, past_pay_out], dim=1)
        else:
            skip = torch.cat([h_ind, features_out], dim=1)
        ind_pre = self.fc_ind_out(skip) + ay_ind_effect
        indicator = torch.sigmoid(ind_pre)

        # Mean head
        h_mean = torch.tanh(self.fc_mean_hidden(h))
        if past_pay_out is not None:
            skip_m = torch.cat([h_mean, features_out, past_pay_out], dim=1)
        else:
            skip_m = torch.cat([h_mean, features_out], dim=1)
        mean = self.fc_mean_out(skip_m) + ay_mean_effect

        return indicator, mean


class GabrielliNN(nn.Module):
    """
    Full Gabrielli (2021) neural network: N_DY subnets with shared embeddings.

    Jointly predicts payment indicators and log-payment means for all DYs.
    """

    def __init__(self, starting_values=None, dropout_rates=None):
        """
        Parameters
        ----------
        starting_values : np.ndarray (N_DY, 2) or None
            Col 0 = mean P(pay>0), Col 1 = mean log(pay|pay>0) per DY.
            Used to initialise output biases.
        dropout_rates : list/array of length N_DY-1, or None
            Dropout rates for past payment info.  Default: 1/[2,3,..,N_DY].
        """
        super().__init__()

        if dropout_rates is None:
            dropout_rates = [1.0 / k for k in range(2, N_DY + 1)]
        self.dropout_rates = dropout_rates

        # ---- Embeddings (dual: input + output) ----
        # Claim type
        self.embed_ct_in = nn.Embedding(EMBED_DIMS["claim_type"], 1)
        self.embed_ct_out = nn.Embedding(EMBED_DIMS["claim_type"], 1)
        # AY
        self.embed_ay_in = nn.Embedding(EMBED_DIMS["AY"], 1)
        self.embed_ay_out = nn.Embedding(EMBED_DIMS["AY"], 1)
        # AQ — now binned notidel (4 categories)
        self.embed_aq_in = nn.Embedding(EMBED_DIMS["AQ"], 1)
        self.embed_aq_out = nn.Embedding(EMBED_DIMS["AQ"], 1)
        # Report delay — (A) now 6 categories (uncapped)
        self.embed_rd_in = nn.Embedding(EMBED_DIMS["report_dy"], 1)
        self.embed_rd_out = nn.Embedding(EMBED_DIMS["report_dy"], 1)

        # (B,D) Linear projections for numeric features -> 1D each
        # (to match the scale of embedding outputs)
        self.fc_numeric_in = nn.Linear(N_NUMERIC_FEATURES, N_NUMERIC_FEATURES)
        self.fc_numeric_out = nn.Linear(N_NUMERIC_FEATURES, N_NUMERIC_FEATURES)

        # Past payment info embeddings (one per DY, DY1..DY{N_DY-1})
        self.embed_pay_in = nn.ModuleList(
            [nn.Embedding(EMBED_DIMS["pay_info"], 1) for _ in range(N_DY - 1)]
        )
        self.embed_pay_out = nn.ModuleList(
            [nn.Embedding(EMBED_DIMS["pay_info"], 1) for _ in range(N_DY - 1)]
        )

        # AY additive effects on outputs (shared across all subnets)
        self.ay_indicator_effect = nn.Linear(1, 1)
        self.ay_mean_effect = nn.Linear(1, 1)
        # Initialise at zero (no calendar-year effect initially)
        nn.init.zeros_(self.ay_indicator_effect.weight)
        nn.init.zeros_(self.ay_indicator_effect.bias)
        nn.init.zeros_(self.ay_mean_effect.weight)
        nn.init.zeros_(self.ay_mean_effect.bias)

        # ---- Shared AY -> hidden layer (paper Section 4.2.4) ----
        # The R code defines AY_embed_input_NN once and reuses it in all subnets.
        # This ensures AY weights train on all accident years across all subnets.
        self.shared_ay_to_hidden = nn.Linear(1, NEURONS[0])

        # ---- Subnets ----
        self.subnets = nn.ModuleList([GabrielliSubnet(t=t) for t in range(1, N_DY + 1)])

        # Assign the shared AY layer to all subnets
        for subnet in self.subnets:
            subnet.fc_ay_to_hidden = self.shared_ay_to_hidden

        # ---- Dropout layers for past payment masking ----
        self.pay_dropouts = nn.ModuleList(
            [nn.Dropout(p=self.dropout_rates[i]) for i in range(N_DY - 1)]
        )

        # ---- Initialise output biases from starting values ----
        if starting_values is not None:
            self._init_biases(starting_values)

    def _init_biases(self, sv):
        """Set initial output biases from data statistics."""
        for t_idx, subnet in enumerate(self.subnets):
            p = float(np.clip(sv[t_idx, 0], 1e-6, 1 - 1e-6))
            # Indicator bias: logit(p)
            with torch.no_grad():
                nn.init.zeros_(subnet.fc_ind_out.weight)
                subnet.fc_ind_out.bias.fill_(math.log(p / (1 - p)))
                # Mean bias: mean log-payment
                nn.init.zeros_(subnet.fc_mean_out.weight)
                subnet.fc_mean_out.bias.fill_(sv[t_idx, 1])

    def forward(
        self,
        claim_type,
        ay,
        aq,
        report_dy,
        numeric_features,
        pay_info,
        time_known,
        time_pred_ind,
        time_pred_pay,
        use_dropout=True,
    ):
        """
        Parameters
        ----------
        claim_type : (B,) int   claim type codes
        ay : (B,) int           accident year codes (0-based)
        aq : (B,) int           binned notidel codes
        report_dy : (B,) int    reporting delay codes (0-5, uncapped)
        numeric_features : (B, N_NUMERIC_FEATURES) float  continuous features
        pay_info : (B, N_DY-1) int  past payment info categories
        time_known : (B, N_DY) float  Time_Known indicators
        time_pred_ind : (B, N_DY) float  Time_Predict_Indicator masks
        time_pred_pay : (B, N_DY) float  Time_Predict_Payment masks

        Returns
        -------
        indicators : (B, N_DY)  payment probabilities
        means : (B, N_DY)       log-payment means
        """
        B = claim_type.size(0)

        # ---- Compute embeddings ----
        # Embedding outputs are (B, 1) since output_dim=1; keep as (B, 1)
        ct_in = self.embed_ct_in(claim_type)  # (B, 1)
        aq_in = self.embed_aq_in(aq)  # (B, 1)
        rd_in = self.embed_rd_in(report_dy)  # (B, 1)
        ay_in = self.embed_ay_in(ay)  # (B, 1)

        ct_out = self.embed_ct_out(claim_type)
        aq_out = self.embed_aq_out(aq)
        rd_out = self.embed_rd_out(report_dy)
        ay_out = self.embed_ay_out(ay)

        # (B,D) Project numeric features
        num_in = self.fc_numeric_in(numeric_features)  # (B, N_NUMERIC_FEATURES)
        num_out = self.fc_numeric_out(numeric_features)  # (B, N_NUMERIC_FEATURES)

        # features_input = concat(ct, aq, rd, numeric) -> (B, N_CLAIM_FEATURES)
        features_in = torch.cat([ct_in, aq_in, rd_in, num_in], dim=1)
        features_out = torch.cat([ct_out, aq_out, rd_out, num_out], dim=1)

        # AY effects
        ay_ind_eff = self.ay_indicator_effect(ay_out)  # (B, 1)
        ay_mean_eff = self.ay_mean_effect(ay_out)  # (B, 1)

        # ---- Compute past payment embeddings ----
        # pay_info[:, k] is the category for DY k+1 (0-indexed)
        pay_embeds_in = []
        pay_embeds_out = []
        for k in range(N_DY - 1):
            pay_embeds_in.append(self.embed_pay_in[k](pay_info[:, k]))  # (B, 1)
            pay_embeds_out.append(self.embed_pay_out[k](pay_info[:, k]))  # (B, 1)

        # ---- Run each subnet ----
        all_indicators = []
        all_means = []

        # Pre-compute a dropout-mask carrier (scalar 1.0, like Gabrielli's dropout embed)
        ones = torch.ones(B, 1, device=claim_type.device)

        for t in range(1, N_DY + 1):
            t_idx = t - 1  # 0-based

            # Build past payment input for this subnet
            if t == 1:
                past_in = None
                past_out = None
            else:
                # DY 1 (index 0): always included without dropout (like Gabrielli's Pay00)
                masked_in = [pay_embeds_in[0]]
                masked_out = [pay_embeds_out[0]]

                # DYs 2..t-1 (indices 1..t-2): apply cascaded dropout + Time_Known
                # Generate all dropout masks for this subnet upfront.
                # For subnet t (our 1-based) = R's j=t-1 (0-based):
                # - Payment at position k (1-indexed) gets k dropout masks
                # - Masks from indices: t-3, t-4, ..., t-3-k+1
                # This matches eq (5.6) in Gabrielli (2021) and the R code.
                if use_dropout and self.training and t >= 3:
                    # Generate dropout masks: need indices t-3 down to 0
                    dropout_masks = []
                    for d_idx in range(t - 3, -1, -1):
                        dropout_masks.append(self.pay_dropouts[d_idx](ones))
                else:
                    dropout_masks = []

                for k in range(1, t - 1):
                    emb_in = pay_embeds_in[k]
                    emb_out = pay_embeds_out[k]

                    # Apply Time_Known mask
                    tk = time_known[:, k : k + 1]  # (B, 1)
                    emb_in = emb_in * tk
                    emb_out = emb_out * tk

                    # Compound cascaded dropout: payment k gets k masks
                    # (outermost first, then progressively deeper)
                    if use_dropout and self.training and t >= 3:
                        for m in range(k):
                            drop_mask = dropout_masks[m]
                            emb_in = emb_in * drop_mask
                            emb_out = emb_out * drop_mask

                    masked_in.append(emb_in)
                    masked_out.append(emb_out)

                past_in = torch.cat(masked_in, dim=1)  # (B, t-1)
                past_out = torch.cat(masked_out, dim=1)

            # Run subnet
            ind, mean = self.subnets[t_idx](
                features_in,
                ay_in,
                past_in,
                features_out,
                past_out,
                ay_ind_eff,
                ay_mean_eff,
            )

            # Mask by Time_Predict
            ind = ind * time_pred_ind[:, t_idx : t_idx + 1]
            mean = mean * time_pred_pay[:, t_idx : t_idx + 1]

            all_indicators.append(ind)
            all_means.append(mean)

        indicators = torch.cat(all_indicators, dim=1)  # (B, N_DY)
        means = torch.cat(all_means, dim=1)  # (B, N_DY)

        return indicators, means

    def freeze_embeddings(self):
        """Freeze all embedding layers (for training step 2).
        Note: fc_numeric_in/out are NOT frozen — they're dense projections."""
        for module in [
            self.embed_ct_in,
            self.embed_ct_out,
            self.embed_ay_in,
            self.embed_ay_out,
            self.embed_aq_in,
            self.embed_aq_out,
            self.embed_rd_in,
            self.embed_rd_out,
        ]:
            for param in module.parameters():
                param.requires_grad = False
        for emb in self.embed_pay_in:
            for param in emb.parameters():
                param.requires_grad = False
        for emb in self.embed_pay_out:
            for param in emb.parameters():
                param.requires_grad = False

    def unfreeze_embeddings(self):
        """Unfreeze all embedding layers."""
        for p in self.parameters():
            p.requires_grad = True
