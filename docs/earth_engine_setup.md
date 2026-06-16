# Google Earth Engine — Student Setup Guide

This is the **one part of the project you have to do yourself**, because it is tied
to your personal Google login. It takes about **15 minutes**, it is **free** for
students/academic use, and you only do it **once**.

Once you finish, paste your project ID into `configs/config.yaml` and the entire
data pipeline becomes live. Until then, everything still runs on synthetic data,
so you are never blocked.

> **Why can't the assistant do this for me?** Creating a Google account, accepting
> Google's Terms of Service, and clicking through the OAuth consent screen are
> actions Google requires a human to perform with their own credentials. The
> assistant has already written **all** the code and config that uses Earth Engine —
> you are just flipping the switch on access.

---

## What you're setting up (the mental model)

Earth Engine runs **on top of Google Cloud**. Three things have to exist:

1. A **Google account** (you probably already have one).
2. A **Google Cloud project** — think of it as a free "workspace" ID that Earth
   Engine bills usage against. For students this is the **free noncommercial** tier.
3. That project **registered for Earth Engine** noncommercial use.

The project's **ID** (something like `ee-yourname-wildfire`) is the single value
you bring back to this repo.

---

## Step 1 — Sign in / create a Google account

Go to <https://accounts.google.com>. Use your **school email** if you have one —
it strengthens the academic eligibility case in Step 4. A personal Gmail works too.

---

## Step 2 — Create a Google Cloud project

1. Open the Cloud project creator:
   <https://console.cloud.google.com/projectcreate>
2. **Project name:** `wildfire-oregon` (or anything).
3. Note the **Project ID** it generates underneath the name. You can edit it —
   make it memorable, e.g. `ee-<yourname>-wildfire`. **Write this ID down.**
   - Project IDs are globally unique and **cannot be changed later**, so pick well.
4. Organization/Location: leave as **No organization** if you're an individual.
5. Click **Create**. Wait for the project to finish provisioning, then make sure
   it is the **selected** project in the blue bar at the top.

> You may be asked to agree to the Google Cloud Terms of Service the first time.
> You do **not** need to enable billing or add a credit card for noncommercial use.

---

## Step 3 — Enable the Earth Engine API

1. With your project selected, open:
   <https://console.cloud.google.com/apis/library/earthengine.googleapis.com>
2. Click **Enable**. (If it says "Manage", it's already enabled — good.)

---

## Step 4 — Register the project for noncommercial Earth Engine use

This is the eligibility questionnaire that grants free access.

1. Open the Earth Engine registration page:
   <https://code.earthengine.google.com/register>
   (or from the docs: <https://developers.google.com/earth-engine/guides/access>)
2. Choose **Use Earth Engine without a Cloud project? → No, use an existing /
   new Cloud project**, and select the project from Step 2.
3. **Choose how you'll use Earth Engine → Unpaid usage (Noncommercial)**.
4. Pick the use case that fits you, e.g. **"Earth Engine trainee / trainer"** or
   **academic research / education**, and answer the short questionnaire
   (role = *Participant* / *Student*, course dates if asked — an estimate is fine).
5. **Choose your plan → Community (free) tier**. Recommended for students and new
   users; no payment.
6. Submit. Approval for noncommercial is typically **instant**.

> ⚠️ Google has been **pausing unregistered projects** — if you ever see a
> "verify your project" banner, it just means redo this questionnaire. Registering
> now avoids that.

---

## Step 5 — Put the project ID into this repo

Open `configs/config.yaml` and set the ID from Step 2:

```yaml
earth_engine:
  project_id: "ee-yourname-wildfire"   # <-- your real ID here
```

(Or, if you prefer not to edit the file, set an environment variable instead —
it overrides the YAML:)

```powershell
# PowerShell, current terminal only
$env:EE_PROJECT = "ee-yourname-wildfire"
```

---

## Step 6 — Authenticate your machine (one-time browser login)

This proves *your computer* is allowed to act as *your* Google account. Run from
the project folder, inside the uv environment:

```powershell
# from C:\Users\Matthew Custom PC\wildfire-oregon-project
uv run earthengine authenticate
```

- A browser window opens. Sign in with the **same** Google account from Step 1.
- Approve the access request.
- It drops a credential file in your user profile
  (`%USERPROFILE%\.config\earthengine\credentials`). **This file is git-ignored —
  never commit it.**

If the browser flow can't open (e.g. headless), use:

```powershell
uv run earthengine authenticate --auth_mode=notebook
```
and paste the verification code it gives you.

---

## Step 7 — Verify it works

Run the built-in check (this script ships with the repo):

```powershell
uv run python scripts/00_check_earth_engine.py
```

Expected output:

```
✅ Earth Engine initialized with project: ee-yourname-wildfire
✅ Test query OK — Oregon MODIS burned-area image count: <some number>
```

If you see that, **you're done** — every ingest script will now pull live data.
Tell the assistant your project ID is set and we'll run the first real sample pull
together.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ee.ee_exception.EEException: Not signed up for Earth Engine` | Redo **Step 4** (registration) with the correct project selected. |
| `... Permission denied on project ...` / `caller does not have permission` | The API isn't enabled (**Step 3**) or the project ID in config is wrong. |
| `Please authorize access...` every run | Re-run **Step 6** (`earthengine authenticate`). |
| Browser didn't open | Use `--auth_mode=notebook` (Step 6). |
| Project verification banner in Cloud console | Re-run the noncommercial questionnaire (**Step 4**). |
| Wrong Google account got authenticated | `uv run earthengine authenticate --force` and sign in with the right one. |

## Reference links

- Earth Engine access overview: <https://developers.google.com/earth-engine/guides/access>
- Noncommercial tiers: <https://developers.google.com/earth-engine/guides/noncommercial_tiers>
- Higher-education resources: <https://developers.google.com/earth-engine/tutorials/edu>
- Friendly walkthrough (Spatial Thoughts): <https://courses.spatialthoughts.com/gee-sign-up.html>
