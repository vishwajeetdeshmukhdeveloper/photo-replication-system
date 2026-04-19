/**
 * Signature Replication System — Frontend Logic
 * ───────────────────────────────────────────────
 * Handles file upload (drag-and-drop + click), API calls,
 * result display, and download functionality.
 */

(() => {
    "use strict";

    // ── DOM References ────────────────────────────────────────────────────
    const dropZone        = document.getElementById("drop-zone");
    const fileInput       = document.getElementById("file-input");
    const previewContainer= document.getElementById("preview-container");
    const previewImage    = document.getElementById("preview-image");
    const clearBtn        = document.getElementById("clear-btn");
    const replicateBtn    = document.getElementById("replicate-btn");
    const btnText         = replicateBtn.querySelector(".btn__text");
    const btnLoader       = replicateBtn.querySelector(".btn__loader");
    const showStepsToggle = document.getElementById("show-steps-toggle");
    const loadingOverlay  = document.getElementById("loading-overlay");
    const resultsSection  = document.getElementById("results-section");
    const resultOriginal  = document.getElementById("result-original");
    const resultReconst   = document.getElementById("result-reconstructed");
    const downloadBtn     = document.getElementById("download-btn");
    const stepsSection    = document.getElementById("steps-section");
    const stepsGrid       = document.getElementById("steps-grid");

    // Metadata elements
    const metaTime     = document.getElementById("meta-time");
    const metaContours = document.getElementById("meta-contours");
    const metaStroke   = document.getElementById("meta-stroke");
    const metaSize     = document.getElementById("meta-size");

    let selectedFile = null;
    let reconstructedBase64 = null;

    // ── Drop Zone Interactions ────────────────────────────────────────────

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    // Drag events
    ["dragenter", "dragover"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // Clear
    clearBtn.addEventListener("click", () => {
        resetSelection();
    });

    // ── File Handling ─────────────────────────────────────────────────────

    function handleFile(file) {
        const validTypes = ["image/jpeg", "image/png", "image/jpg"];
        if (!validTypes.includes(file.type)) {
            showError("Please select a JPG or PNG image.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError("File is too large. Maximum size is 10 MB.");
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.classList.remove("hidden");
            dropZone.style.display = "none";
            replicateBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetSelection() {
        selectedFile = null;
        reconstructedBase64 = null;
        fileInput.value = "";
        previewContainer.classList.add("hidden");
        dropZone.style.display = "";
        replicateBtn.disabled = true;
        resultsSection.classList.add("hidden");
        loadingOverlay.classList.add("hidden");
    }

    // ── Replicate Request ─────────────────────────────────────────────────

    replicateBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        const showSteps = showStepsToggle.checked;
        const endpoint = showSteps ? "/api/replicate-steps" : "/api/replicate";

        // UI: loading state
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        replicateBtn.disabled = true;
        loadingOverlay.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const res  = await fetch(endpoint, { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok || !data.success) {
                throw new Error(data.detail || "Processing failed");
            }

            displayResults(data, showSteps);

        } catch (err) {
            showError(err.message || "An unexpected error occurred.");
        } finally {
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
            replicateBtn.disabled = false;
            loadingOverlay.classList.add("hidden");
        }
    });

    // ── Display Results ───────────────────────────────────────────────────

    function displayResults(data, showSteps) {
        // Original preview
        resultOriginal.src = previewImage.src;

        // Reconstructed image
        const finalImg = data.final_image || data.reconstructed_image;
        reconstructedBase64 = finalImg;
        resultReconst.src = `data:image/png;base64,${finalImg}`;

        // Metadata
        const meta = data.metadata || {};
        metaTime.textContent     = `${data.processing_time_seconds}s`;
        metaContours.textContent = meta.contour_count ?? "—";
        metaStroke.textContent   = meta.stroke_width_mean ? `${meta.stroke_width_mean}px` : "—";
        metaSize.textContent     = meta.output_shape ? `${meta.output_shape[1]}×${meta.output_shape[0]}` : "—";

        // Processing steps
        stepsGrid.innerHTML = "";
        if (showSteps && data.steps) {
            const stepEntries = Object.entries(data.steps).sort(([a], [b]) => a.localeCompare(b));
            stepEntries.forEach(([name, b64], idx) => {
                const card = document.createElement("div");
                card.className = "step-card";
                card.style.animationDelay = `${idx * 0.06}s`;

                const label = name
                    .replace(/^\d+_/, "")
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (c) => c.toUpperCase());

                card.innerHTML = `
                    <img class="step-card__image" src="data:image/png;base64,${b64}" alt="${label}">
                    <div class="step-card__label">${label}</div>
                `;
                stepsGrid.appendChild(card);
            });
            stepsSection.classList.remove("hidden");
        } else {
            stepsSection.classList.add("hidden");
        }

        resultsSection.classList.remove("hidden");

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Download ──────────────────────────────────────────────────────────

    downloadBtn.addEventListener("click", () => {
        if (!reconstructedBase64) return;

        const link = document.createElement("a");
        link.href = `data:image/png;base64,${reconstructedBase64}`;
        link.download = "reconstructed_signature.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // ── Error Toast ───────────────────────────────────────────────────────

    function showError(msg) {
        // Simple alert for now; could replace with a toast component
        alert(`⚠ ${msg}`);
    }

})();
