const degreeLanguageSelect = document.getElementById("degree_language");
const englishFields = document.querySelector(".english-fields");
const chineseFields = document.querySelector(".chinese-fields");
const englishScoreInput = document.getElementById("english_score");
const englishTestSelect = document.getElementById("english_test_type");

function toggleLanguageSections(value) {
    if (value === "english_taught") {
        englishFields.hidden = false;
        chineseFields.hidden = true;
        englishFields.querySelectorAll("input, select").forEach(el => el.required = true);
        chineseFields.querySelectorAll("select").forEach(el => el.required = false);
    } else {
        englishFields.hidden = true;
        chineseFields.hidden = false;
        englishFields.querySelectorAll("input, select").forEach(el => el.required = false);
        chineseFields.querySelectorAll("select").forEach(el => el.required = true);
    }
}

toggleLanguageSections(degreeLanguageSelect.value);
degreeLanguageSelect.addEventListener("change", (event) => {
    toggleLanguageSections(event.target.value);
});

const englishScoreRanges = {
    toefl: { min: 0, max: 120, step: 1, placeholder: "Score (0-120)" },
    ielts: { min: 0, max: 9, step: 0.5, placeholder: "Score (0-9)" },
    duolingo: { min: 10, max: 160, step: 1, placeholder: "Score (10-160)" },
};

function applyEnglishScoreConstraints(testType) {
    if (!englishScoreInput) return;
    const config = englishScoreRanges[testType] || englishScoreRanges.toefl;

    englishScoreInput.min = config.min;
    englishScoreInput.max = config.max;
    englishScoreInput.step = config.step;
    englishScoreInput.placeholder = config.placeholder;

    const currentValue = parseFloat(englishScoreInput.value);
    if (!Number.isNaN(currentValue)) {
        const clamped = Math.min(config.max, Math.max(config.min, currentValue));
        englishScoreInput.value = clamped;
    }
}

if (englishTestSelect) {
    applyEnglishScoreConstraints(englishTestSelect.value);
    englishTestSelect.addEventListener("change", (event) => {
        applyEnglishScoreConstraints(event.target.value);
    });
}

const radarLabels = [
    "GPA",
    "Math/Physics",
    "Research",
    "Publications",
    "Recommendation",
    "Interview",
    "Language Pass"
];

const radarCanvas = document.getElementById("radarCanvas");
let radarChart;

function renderRadarChart(dataPoints) {
    if (!radarCanvas) return;

    if (radarChart) {
        radarChart.destroy();
    }

    radarChart = new Chart(radarCanvas, {
        type: "radar",
        data: {
            labels: radarLabels,
            datasets: [{
                label: "Scores(Normalized)",
                data: dataPoints,
                fill: true,
                backgroundColor: "rgba(31, 111, 235, 0.15)",
                borderColor: "rgba(31, 111, 235, 0.6)",
                pointBackgroundColor: "rgba(31, 111, 235, 0.9)",
                pointBorderColor: "#fff",
                pointHoverBackgroundColor: "#fff",
                pointHoverBorderColor: "rgba(31, 111, 235, 1)",
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    suggestedMax: 1,
                    ticks: {
                        backdropColor: "transparent",
                        color: "var(--text-secondary)",
                        stepSize: 0.2
                    },
                    grid: {
                        color: "rgba(15, 23, 42, 0.1)"
                    },
                    angleLines: {
                        color: "rgba(15, 23, 42, 0.08)"
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

const predictionTimestampEl = document.getElementById("prediction-timestamp");
const predictionResultEl = document.getElementById("prediction-result");

function renderPredictionCards(predictionPayload) {
    if (!predictionResultEl) return;

    if (!predictionPayload || Object.keys(predictionPayload).length === 0) {
        predictionResultEl.innerHTML = "<p class=\"placeholder\">Submit applicant details to see admission and scholarship predictions here.</p>";
        return;
    }

    const fragment = document.createDocumentFragment();
    Object.entries(predictionPayload).forEach(([key, value]) => {
        const card = document.createElement("div");
        card.className = "prediction-card";

        const labelEl = document.createElement("span");
        labelEl.textContent = key.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase());

        const valueEl = document.createElement("strong");
        valueEl.textContent = value;

        card.append(labelEl, valueEl);
        fragment.appendChild(card);
    });

    predictionResultEl.innerHTML = "";
    predictionResultEl.appendChild(fragment);
}

const initialRadarData = Array.isArray(window.initialRadarData) ? window.initialRadarData : [0, 0, 0, 0, 0, 0, 0];
renderRadarChart(initialRadarData);

if (window.initialPredictions) {
    renderPredictionCards(window.initialPredictions);
}

if (window.initialTimestamp && predictionTimestampEl) {
    predictionTimestampEl.textContent = window.initialTimestamp;
}
