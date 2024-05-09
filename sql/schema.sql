CREATE TABLE "risk_rate" (
"index" INTEGER,
  "rate" REAL,
  "date" DATE
);

CREATE INDEX "ix_risk_rate_index"ON "risk_rate" ("index");

CREATE TABLE "stock_history" (
"Date" TIMESTAMP,
  "Open" REAL,
  "High" REAL,
  "Low" REAL,
  "Close" REAL,
  "Adj Close" REAL,
  "Volume" INTEGER,
  "ticker" TEXT
);

CREATE INDEX "ix_stock_history_Date" ON "stock_history" ("Date");
CREATE INDEX "ix_stock_history_Date-Ticker" ON "stock_history" ("Date", "ticker");


CREATE TABLE "mcap_stock" (
"index" INTEGER,
  "industry" TEXT,
  "market_cap" REAL,
  "ticker" TEXT
);
CREATE INDEX "ix_mcap_stock_index" ON "mcap_stock" ("index");
