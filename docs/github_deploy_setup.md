# Deploy the live site + automatic weekly updates

This sets up the site to **deploy itself to GitHub Pages** on every change, and
to **auto-refresh predictions vs. actuals every Monday** during fire season — no
manual steps once it's wired up.

Once done you get a public URL like `https://<your-user>.github.io/<repo>/` that
stays current on its own.

You only have to do this **once**. Total: ~20 minutes. Everything is free for
public repos + Earth Engine noncommercial.

---

## What you'll set up

1. A GitHub repository for this project (public).
2. GitHub Pages turned on, deployed by Actions (`.github/workflows/pages.yml`).
3. A Google Cloud **service account** with Earth Engine access — so the headless
   GitHub Actions runner can pull MODIS data on your behalf (you can't share your
   personal `earthengine authenticate` token).
4. The service-account key + your EE project id stored as GitHub **Secrets**.
5. The weekly schedule (`.github/workflows/update.yml`) starts running automatically.

> Why a service account? Earth Engine's interactive auth is browser-OAuth — it
> can't run in a headless CI job. A service account is the standard solution and
> is what Google's own docs recommend for this exact case.

---

## Step 1 — Create a GitHub repo and push the code

Skip this if your repo already exists.

```powershell
# from the project root, after `gh auth login` once
gh repo create wildfire-oregon-project --public --source=. --remote=origin --push
```

(Or create it in the GitHub UI and `git remote add origin … && git push -u origin main`.)

---

## Step 2 — Enable GitHub Pages

In the repo on GitHub: **Settings → Pages → Source: GitHub Actions**.

That's it — the `pages.yml` workflow will deploy on the next push.

---

## Step 3 — Create the Earth Engine service account

1. Open the Service Accounts page for your Cloud project:
   <https://console.cloud.google.com/iam-admin/serviceaccounts>
   (Make sure your **wildfire** project is selected, top blue bar.)

2. Click **Create service account**.
   - **Name:** `wildfire-ee-bot`
   - **Description:** "GitHub Actions runner for Earth Engine pulls"
   - Click **Create and continue**.

3. **Grant this service account access** — give it the role:
   - **Earth Engine Resource Viewer** (sufficient for queries)
   - (Optional, only if you ever export tables to Drive from CI:
     also add **Service Usage Consumer**.)

   Click **Continue** then **Done**.

4. Open the service account you just created. Go to the **Keys** tab.
5. **Add Key → Create new key → JSON**. A JSON file downloads —
   keep it; we'll use it in Step 4.

### Register the service account with Earth Engine

The Earth Engine Cloud project needs to know this service account is allowed.
Open <https://code.earthengine.google.com/register> and:

- Make sure your `wildfire-…` project is selected at the top.
- In **Service accounts**, paste the service-account email
  (looks like `wildfire-ee-bot@<project>.iam.gserviceaccount.com`) and click
  **Register**.

---

## Step 4 — Add the GitHub secrets

The JSON key shouldn't be committed. Put it in GitHub Secrets instead.

1. **Encode the JSON to base64** so it's a single line:

```powershell
# PowerShell (Windows)
$json = Get-Content "C:\path\to\downloaded\service-account-key.json" -Raw
[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($json))
```

```bash
# bash (macOS/Linux)
base64 -w0 service-account-key.json
```

Copy the long base64 string.

2. In your repo on GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
   Add **two** secrets:

   | Name | Value |
   |---|---|
   | `EE_SERVICE_ACCOUNT_KEY` | the base64 string from above |
   | `EE_PROJECT` | your EE project id, e.g. `wildfire-prediction-499606` |

3. Delete the downloaded JSON file from your computer; the secret is the canonical copy.

---

## Step 5 — Trigger the first run

You don't have to wait until Monday — kick it off manually to make sure
everything works:

1. In the repo: **Actions** tab → **Update predictions vs. actuals** → **Run workflow** → **Run workflow**.
2. Watch the run. It should:
   - Install uv + Python + project deps.
   - Initialize Earth Engine with the service account.
   - Pull the most recent MODIS labels for the target year.
   - Update `site/data/predictions.json` and `predictions.js`.
   - Commit the updated files back to `main`.
3. The push to `main` automatically triggers the **Deploy site to GitHub Pages** workflow,
   which rebuilds and deploys the site.
4. Visit your Pages URL — the Predictions Tracker tab now shows real "Actual" numbers
   alongside "Predicted", with per-cell hit/miss badges.

After this first manual run, the schedule (`cron: "0 9 * 5-11 1"`) takes over and
runs automatically every Monday morning during fire season.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Earth Engine failed to initialize` in the Action log | The service account isn't registered with EE (Step 3 end). Re-do the registration page. |
| `403 Permission denied` on the EE call | The service account needs **Earth Engine Resource Viewer** role on the project (Step 3.3). |
| `EE_SERVICE_ACCOUNT_KEY is set but couldn't be decoded as base64 JSON` | Re-run the base64 command; check you copied the **whole** string with no newlines. |
| Pages deploy fails: "Resource not accessible" | In Settings → Pages, source must be **GitHub Actions**, not "Deploy from branch". |
| Schedule isn't firing | GitHub disables scheduled workflows in repos with no recent activity. A single push to `main` re-enables it. |

---

## What runs where, summarized

| Trigger | Workflow | What it does |
|---|---|---|
| Push to `main` (anything in `site/`) | `pages.yml` | Deploys current `site/` to Pages |
| Manual ▶ in Actions tab | `update.yml` | Pulls fresh MODIS, refreshes predictions, commits |
| Cron, Mondays 09:00 UTC, May–Nov | `update.yml` | Same — fully automatic |

That's it. Once the secrets are in, you don't touch anything.
