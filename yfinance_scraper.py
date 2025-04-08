import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration with custom theme
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4B5563;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-positive {
        color: #047857;
    }
    .metric-negative {
        color: #DC2626;
    }
    .metric-neutral {
        color: #4B5563;
    }
    .stExpander {
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        border-radius: 0.5rem !important;
    }
    div[data-testid="stExpander"] > div[role="button"] > div[data-testid="stMarkdown"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for company selection and time period
with st.sidebar:
    st.markdown("## 🔍 Company Selection")
    ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL")
    
    # Add a search button
    search_button = st.button("Analyze Company", use_container_width=True)
    
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    
    # Time period selection for historical data
    time_period = st.selectbox(
        "Historical Data Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "Max"],
        index=3
    )
    
    # Currency display option
    currency = st.selectbox("Display Currency", ["USD ($)", "EUR (€)", "GBP (£)"], index=0)
    currency_symbol = currency.split(" ")[1].strip("()")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This dashboard provides comprehensive financial analysis for publicly traded companies using data from Yahoo Finance.
    
    **Metrics included:**
    - Profitability
    - Valuation
    - Solvency & Liquidity
    - Growth
    - Dividends
    """)

# Main content
st.markdown('<h1 class="main-header">📊 Financial Analysis Dashboard</h1>', unsafe_allow_html=True)

# Function to get valid row from dataframe
def get_first_valid_row(possibles, df):
    for key in possibles:
        if key in df.index:
            return key
    return None

# Function to format numbers with appropriate suffixes
def format_number(num):
    if num is None:
        return "N/A"
    
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num/1_000_000_000_000:.2f}T"
    elif abs_num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

# Function to create a metric card
def metric_card(label, value, prefix="", suffix="", delta=None, delta_suffix="%"):
    delta_html = ""
    if delta is not None:
        if delta > 0:
            delta_html = f'<span class="metric-positive">▲ {delta:.2f}{delta_suffix}</span>'
        elif delta < 0:
            delta_html = f'<span class="metric-negative">▼ {abs(delta):.2f}{delta_suffix}</span>'
        else:
            delta_html = f'<span class="metric-neutral">◆ {delta:.2f}{delta_suffix}</span>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """

# Function to get time period in days
def get_period_days(period_str):
    if period_str == "1 Month":
        return 30
    elif period_str == "3 Months":
        return 90
    elif period_str == "6 Months":
        return 180
    elif period_str == "1 Year":
        return 365
    elif period_str == "3 Years":
        return 365 * 3
    elif period_str == "5 Years":
        return 365 * 5
    else:  # Max
        return 365 * 20  # Arbitrary large number

# Main analysis function
def analyze_company(ticker_symbol):
    try:
        # Display a loading spinner
        with st.spinner(f"Analyzing {ticker_symbol}..."):
            # Get ticker data
            ticker = yf.Ticker(ticker_symbol)
            
            # Get company info
            info = ticker.info
            company_name = info.get("longName", ticker_symbol)
            
            # Display company header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"## {company_name} ({ticker_symbol})")
                sector = info.get("sector", "N/A")
                industry = info.get("industry", "N/A")
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
            
            with col2:
                # Current price and daily change
                current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
                previous_close = info.get("previousClose", 0)
                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100 if previous_close else 0
                
                price_color = "green" if price_change >= 0 else "red"
                change_symbol = "▲" if price_change >= 0 else "▼"
                
                st.markdown(f"""
                <div style="text-align: right;">
                    <div style="font-size: 2rem; font-weight: 700; color: #1E3A8A;">{currency_symbol}{current_price:.2f}</div>
                    <div style="font-size: 1.1rem; color: {price_color};">
                        {change_symbol} {currency_symbol}{abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Get financial data
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Use the most recent column available
            latest_col = income_stmt.columns[0] if not income_stmt.empty else None
            
            # Create tabs for different analysis sections
            tabs = st.tabs(["📈 Overview", "💰 Profitability", "💹 Valuation", "🔐 Solvency", "📊 Growth", "💵 Dividends"])
            
            # Tab 1: Overview
            with tabs[0]:
                # Stock price chart
                st.markdown('<h2 class="section-header">Stock Price History</h2>', unsafe_allow_html=True)
                
                # Get historical data
                period_days = get_period_days(time_period)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=period_days)
                
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Create interactive chart with Plotly
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1E3A8A', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{company_name} Stock Price ({time_period})",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({currency_symbol})",
                        height=500,
                        template="plotly_white",
                        hovermode="x unified",
                        xaxis=dict(
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No historical data available for the selected time period.")
                
                # Key metrics overview
                st.markdown('<h2 class="section-header">Key Metrics Overview</h2>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Market cap
                market_cap = info.get("marketCap", None)
                with col1:
                    st.markdown(
                        metric_card("Market Cap", format_number(market_cap), prefix=currency_symbol),
                        unsafe_allow_html=True
                    )
                
                # P/E Ratio
                pe_ratio = info.get("trailingPE", None)
                with col2:
                    st.markdown(
                        metric_card("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A"),
                        unsafe_allow_html=True
                    )
                
                # ROE
                if latest_col is not None:
                    net_income_key = get_first_valid_row(["Net Income", "NetIncome"], income_stmt)
                    equity_key = get_first_valid_row(["Total Stockholder Equity", "Common Stock Equity", "Total Equity", "Stockholders Equity"], balance_sheet)
                    
                    net_income = income_stmt.loc[net_income_key, latest_col] if net_income_key else None
                    total_equity = balance_sheet.loc[equity_key, latest_col] if equity_key else None
                    
                    roe = (net_income / total_equity) * 100 if net_income and total_equity else None
                    
                    with col3:
                        st.markdown(
                            metric_card("Return on Equity", f"{roe:.2f}" if roe else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                
                # Dividend Yield
                dividend_yield = info.get("dividendYield", None)
                with col4:
                    st.markdown(
                        metric_card(
                            "Dividend Yield", 
                            f"{dividend_yield*100:.2f}" if dividend_yield else "N/A", 
                            suffix="%"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Company description
                st.markdown('<h2 class="section-header">Company Description</h2>', unsafe_allow_html=True)
                description = info.get("longBusinessSummary", "No description available.")
                st.markdown(f"<div style='text-align: justify;'>{description}</div>", unsafe_allow_html=True)
            
            # Tab 2: Profitability
            with tabs[1]:
                st.markdown('<h2 class="section-header">Profitability Metrics</h2>', unsafe_allow_html=True)
                
                if latest_col is not None:
                    # Get necessary data
                    net_income_key = get_first_valid_row(["Net Income", "NetIncome"], income_stmt)
                    revenue_key = get_first_valid_row(["Total Revenue", "Revenue"], income_stmt)
                    equity_key = get_first_valid_row(["Total Stockholder Equity", "Common Stock Equity", "Total Equity", "Stockholders Equity"], balance_sheet)
                    assets_key = get_first_valid_row(["Total Assets"], balance_sheet)
                    
                    net_income = income_stmt.loc[net_income_key, latest_col] if net_income_key else None
                    revenue = income_stmt.loc[revenue_key, latest_col] if revenue_key else None
                    total_equity = balance_sheet.loc[equity_key, latest_col] if equity_key else None
                    total_assets = balance_sheet.loc[assets_key, latest_col] if assets_key else None
                    
                    # Calculate metrics
                    net_profit_margin = (net_income / revenue) * 100 if revenue and net_income else None
                    roe = (net_income / total_equity) * 100 if total_equity and net_income else None
                    roa = (net_income / total_assets) * 100 if total_assets and net_income else None
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            metric_card("Net Profit Margin", f"{net_profit_margin:.2f}" if net_profit_margin else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            metric_card("Return on Equity (ROE)", f"{roe:.2f}" if roe else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            metric_card("Return on Assets (ROA)", f"{roa:.2f}" if roa else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    # Profitability over time (if multiple periods available)
                    if len(income_stmt.columns) > 1:
                        st.markdown('<h2 class="section-header">Profitability Trends</h2>', unsafe_allow_html=True)
                        
                        # Calculate metrics for each period
                        periods = []
                        profit_margins = []
                        roes = []
                        roas = []
                        
                        for col in income_stmt.columns[:4]:  # Use up to 4 most recent periods
                            period_net_income = income_stmt.loc[net_income_key, col] if net_income_key else None
                            period_revenue = income_stmt.loc[revenue_key, col] if revenue_key else None
                            
                            # Find corresponding balance sheet data (might be different dates)
                            closest_bs_col = min(balance_sheet.columns, key=lambda x: abs((x - col).days))
                            period_equity = balance_sheet.loc[equity_key, closest_bs_col] if equity_key else None
                            period_assets = balance_sheet.loc[assets_key, closest_bs_col] if assets_key else None
                            
                            # Calculate metrics
                            period_profit_margin = (period_net_income / period_revenue) * 100 if period_revenue and period_net_income else None
                            period_roe = (period_net_income / period_equity) * 100 if period_equity and period_net_income else None
                            period_roa = (period_net_income / period_assets) * 100 if period_assets and period_net_income else None
                            
                            # Add to lists
                            periods.append(col.strftime('%Y-%m'))
                            profit_margins.append(period_profit_margin)
                            roes.append(period_roe)
                            roas.append(period_roa)
                        
                        # Create dataframe for chart
                        trend_data = pd.DataFrame({
                            'Period': periods,
                            'Net Profit Margin (%)': profit_margins,
                            'ROE (%)': roes,
                            'ROA (%)': roas
                        })
                        
                        # Create chart
                        fig = px.line(
                            trend_data.melt(id_vars=['Period'], var_name='Metric', value_name='Value'),
                            x='Period',
                            y='Value',
                            color='Metric',
                            markers=True,
                            title='Profitability Metrics Over Time',
                            template='plotly_white'
                        )
                        
                        fig.update_layout(
                            xaxis_title='',
                            yaxis_title='Percentage (%)',
                            legend_title='',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Financial statements
                    with st.expander("📄 Detailed Financial Statements"):
                        st.markdown(f"**Latest Period:** {latest_col.strftime('%Y-%m-%d')}")
                        
                        st.markdown("### Income Statement")
                        st.dataframe(income_stmt.style.format("${:,.0f}"))
                        
                        st.markdown("### Balance Sheet")
                        st.dataframe(balance_sheet.style.format("${:,.0f}"))
                else:
                    st.warning("No financial data available for this company.")
            
            # Tab 3: Valuation
            with tabs[2]:
                st.markdown('<h2 class="section-header">Valuation Metrics</h2>', unsafe_allow_html=True)
                
                # Get valuation metrics
                market_cap = info.get("marketCap", None)
                pe_ratio = info.get("trailingPE", None)
                forward_pe = info.get("forwardPE", None)
                pb_ratio = info.get("priceToBook", None)
                ps_ratio = info.get("priceToSalesTrailing12Months", None)
                ev_to_ebitda = info.get("enterpriseToEbitda", None)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        metric_card("Market Cap", format_number(market_cap), prefix=currency_symbol),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        metric_card("P/E Ratio (TTM)", f"{pe_ratio:.2f}" if pe_ratio else "N/A"),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        metric_card("Forward P/E", f"{forward_pe:.2f}" if forward_pe else "N/A"),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        metric_card("Price/Book Ratio", f"{pb_ratio:.2f}" if pb_ratio else "N/A"),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        metric_card("Price/Sales Ratio", f"{ps_ratio:.2f}" if ps_ratio else "N/A"),
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        metric_card("EV/EBITDA", f"{ev_to_ebitda:.2f}" if ev_to_ebitda else "N/A"),
                        unsafe_allow_html=True
                    )
                
                # Valuation comparison with industry
                st.markdown('<h2 class="section-header">Valuation Comparison</h2>', unsafe_allow_html=True)
                
                # Get industry averages (these would normally come from a database or API)
                # For demonstration, we'll use placeholder values
                industry_pe = info.get("industryPE", 20)  # Placeholder
                industry_pb = info.get("industryPB", 3)   # Placeholder
                industry_ps = info.get("industryPS", 2)   # Placeholder
                
                # Create comparison chart
                comparison_data = {
                    'Metric': ['P/E Ratio', 'P/B Ratio', 'P/S Ratio'],
                    f'{company_name}': [pe_ratio or 0, pb_ratio or 0, ps_ratio or 0],
                    'Industry Average': [industry_pe, industry_pb, industry_ps]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                
                fig = px.bar(
                    df_comparison.melt(id_vars=['Metric'], var_name='Entity', value_name='Value'),
                    x='Metric',
                    y='Value',
                    color='Entity',
                    barmode='group',
                    title='Valuation Metrics Comparison',
                    template='plotly_white'
                )
                
                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='Value',
                    legend_title='',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Valuation details
                with st.expander("📄 Detailed Valuation Information"):
                    st.markdown("### Key Valuation Components")
                    
                    price = info.get("currentPrice", None)
                    eps = info.get("trailingEps", None)
                    book_value = info.get("bookValue", None)
                    sales_per_share = info.get("revenuePerShare", None)
                    
                    st.markdown(f"**Current Price:** {currency_symbol}{price:.2f}" if price else "**Current Price:** N/A")
                    st.markdown(f"**Earnings Per Share (TTM):** {currency_symbol}{eps:.2f}" if eps else "**Earnings Per Share (TTM):** N/A")
                    st.markdown(f"**Book Value Per Share:** {currency_symbol}{book_value:.2f}" if book_value else "**Book Value Per Share:** N/A")
                    st.markdown(f"**Sales Per Share:** {currency_symbol}{sales_per_share:.2f}" if sales_per_share else "**Sales Per Share:** N/A")
                    
                    st.markdown("### Market Valuation")
                    st.markdown(f"**Market Cap:** {currency_symbol}{format_number(market_cap)}" if market_cap else "**Market Cap:** N/A")
                    
                    enterprise_value = info.get("enterpriseValue", None)
                    st.markdown(f"**Enterprise Value:** {currency_symbol}{format_number(enterprise_value)}" if enterprise_value else "**Enterprise Value:** N/A")
            
            # Tab 4: Solvency & Liquidity
            with tabs[3]:
                st.markdown('<h2 class="section-header">Solvency & Liquidity Metrics</h2>', unsafe_allow_html=True)
                
                if latest_col is not None:
                    # Get necessary data
                    total_debt = info.get("totalDebt", None)
                    
                    # Try to get from balance sheet if not in info
                    if total_debt is None and not balance_sheet.empty:
                        debt_key = get_first_valid_row(["Total Debt", "Long Term Debt"], balance_sheet)
                        if debt_key:
                            total_debt = balance_sheet.loc[debt_key, latest_col]
                    
                    total_equity = None
                    if not balance_sheet.empty:
                        equity_key = get_first_valid_row(["Total Stockholder Equity", "Common Stock Equity", "Total Equity", "Stockholders Equity"], balance_sheet)
                        if equity_key:
                            total_equity = balance_sheet.loc[equity_key, latest_col]
                    
                    current_assets = None
                    if not balance_sheet.empty:
                        ca_key = get_first_valid_row(["Total Current Assets", "Current Assets"], balance_sheet)
                        if ca_key:
                            current_assets = balance_sheet.loc[ca_key, latest_col]
                    
                    current_liabilities = None
                    if not balance_sheet.empty:
                        cl_key = get_first_valid_row(["Total Current Liabilities", "Current Liabilities"], balance_sheet)
                        if cl_key:
                            current_liabilities = balance_sheet.loc[cl_key, latest_col]
                    
                    # Calculate metrics
                    debt_to_equity = (total_debt / total_equity) if total_debt and total_equity else None
                    current_ratio = (current_assets / current_liabilities) if current_assets and current_liabilities else None
                    
                    # Get quick ratio components
                    quick_assets = None
                    if not balance_sheet.empty:
                        cash_key = get_first_valid_row(["Cash And Cash Equivalents", "Cash", "Cash and Short Term Investments"], balance_sheet)
                        receivables_key = get_first_valid_row(["Net Receivables", "Accounts Receivable"], balance_sheet)
                        
                        cash = balance_sheet.loc[cash_key, latest_col] if cash_key else 0
                        receivables = balance_sheet.loc[receivables_key, latest_col] if receivables_key else 0
                        
                        quick_assets = cash + receivables
                    
                    quick_ratio = (quick_assets / current_liabilities) if quick_assets and current_liabilities else None
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            metric_card("Debt-to-Equity", f"{debt_to_equity:.2f}" if debt_to_equity else "N/A"),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            metric_card("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A"),
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            metric_card("Quick Ratio", f"{quick_ratio:.2f}" if quick_ratio else "N/A"),
                            unsafe_allow_html=True
                        )
                    
                    # Debt composition
                    st.markdown('<h2 class="section-header">Debt Composition</h2>', unsafe_allow_html=True)
                    
                    # Get debt components
                    long_term_debt = None
                    short_term_debt = None
                    
                    if not balance_sheet.empty:
                        ltd_key = get_first_valid_row(["Long Term Debt"], balance_sheet)
                        std_key = get_first_valid_row(["Short Term Debt", "Current Debt"], balance_sheet)
                        
                        long_term_debt = balance_sheet.loc[ltd_key, latest_col] if ltd_key else None
                        short_term_debt = balance_sheet.loc[std_key, latest_col] if std_key else None
                    
                    # Create pie chart for debt composition
                    if long_term_debt or short_term_debt:
                        debt_data = {
                            'Category': [],
                            'Amount': []
                        }
                        
                        if long_term_debt:
                            debt_data['Category'].append('Long-term Debt')
                            debt_data['Amount'].append(long_term_debt)
                        
                        if short_term_debt:
                            debt_data['Category'].append('Short-term Debt')
                            debt_data['Amount'].append(short_term_debt)
                        
                        df_debt = pd.DataFrame(debt_data)
                        
                        fig = px.pie(
                            df_debt,
                            values='Amount',
                            names='Category',
                            title='Debt Composition',
                            template='plotly_white',
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Debt composition data not available.")
                    
                    # Solvency details
                    with st.expander("📄 Detailed Solvency Information"):
                        st.markdown("### Balance Sheet Components")
                        
                        st.markdown(f"**Total Debt:** {currency_symbol}{format_number(total_debt)}" if total_debt else "**Total Debt:** N/A")
                        st.markdown(f"**Total Equity:** {currency_symbol}{format_number(total_equity)}" if total_equity else "**Total Equity:** N/A")
                        st.markdown(f"**Current Assets:** {currency_symbol}{format_number(current_assets)}" if current_assets else "**Current Assets:** N/A")
                        st.markdown(f"**Current Liabilities:** {currency_symbol}{format_number(current_liabilities)}" if current_liabilities else "**Current Liabilities:** N/A")
                        
                        if not balance_sheet.empty:
                            st.markdown("### Full Balance Sheet")
                            st.dataframe(balance_sheet.style.format("${:,.0f}"))
                else:
                    st.warning("No financial data available for this company.")
            
            # Tab 5: Growth
            with tabs[4]:
                st.markdown('<h2 class="section-header">Growth Metrics</h2>', unsafe_allow_html=True)
                
                # Check if we have enough data for growth calculations
                if not income_stmt.empty and len(income_stmt.columns) >= 2:
                    latest_period = income_stmt.columns[0]
                    previous_period = income_stmt.columns[1]
                    
                    # Get revenue data
                    revenue_key = get_first_valid_row(["Total Revenue", "Revenue"], income_stmt)
                    current_revenue = income_stmt.loc[revenue_key, latest_period] if revenue_key else None
                    previous_revenue = income_stmt.loc[revenue_key, previous_period] if revenue_key else None
                    
                    # Get net income data
                    net_income_key = get_first_valid_row(["Net Income", "NetIncome"], income_stmt)
                    current_net_income = income_stmt.loc[net_income_key, latest_period] if net_income_key else None
                    previous_net_income = income_stmt.loc[net_income_key, previous_period] if net_income_key else None
                    
                    # Calculate growth rates
                    revenue_growth = ((current_revenue / previous_revenue) - 1) * 100 if current_revenue and previous_revenue else None
                    net_income_growth = ((current_net_income / previous_net_income) - 1) * 100 if current_net_income and previous_net_income else None
                    
                    # EPS growth
                    current_eps = info.get("trailingEps", None)
                    previous_eps = None
                    
                    # Try to calculate previous EPS
                    shares_outstanding = info.get("sharesOutstanding", None)
                    if shares_outstanding and previous_net_income:
                        previous_eps = previous_net_income / shares_outstanding
                    
                    eps_growth = ((current_eps / previous_eps) - 1) * 100 if current_eps and previous_eps else None
                    
                    # Display growth metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(
                            metric_card("Revenue Growth", f"{revenue_growth:.2f}" if revenue_growth else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            metric_card("Net Income Growth", f"{net_income_growth:.2f}" if net_income_growth else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            metric_card("EPS Growth", f"{eps_growth:.2f}" if eps_growth else "N/A", suffix="%"),
                            unsafe_allow_html=True
                        )
                    
                    # Historical growth chart
                    st.markdown('<h2 class="section-header">Historical Growth Trends</h2>', unsafe_allow_html=True)
                    
                    # Get data for multiple periods
                    periods = []
                    revenues = []
                    net_incomes = []
                    
                    for col in income_stmt.columns[:5]:  # Use up to 5 most recent periods
                        period_revenue = income_stmt.loc[revenue_key, col] if revenue_key else None
                        period_net_income = income_stmt.loc[net_income_key, col] if net_income_key else None
                        
                        periods.append(col.strftime('%Y-%m'))
                        revenues.append(period_revenue)
                        net_incomes.append(period_net_income)
                    
                    # Create dataframe for chart
                    trend_data = pd.DataFrame({
                        'Period': periods,
                        'Revenue': revenues,
                        'Net Income': net_incomes
                    })
                    
                    # Reverse order to show oldest to newest
                    trend_data = trend_data.iloc[::-1].reset_index(drop=True)
                    
                    # Create chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=trend_data['Period'],
                        y=trend_data['Revenue'],
                        name='Revenue',
                        marker_color='#1E3A8A'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_data['Period'],
                        y=trend_data['Net Income'],
                        name='Net Income',
                        mode='lines+markers',
                        line=dict(color='#10B981', width=3)
                    ))
                    
                    fig.update_layout(
                        title='Revenue and Net Income Trends',
                        xaxis_title='',
                        yaxis_title=f'Amount ({currency_symbol})',
                        template='plotly_white',
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Format y-axis to show abbreviated values
                    fig.update_yaxes(tickformat=".2s")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Growth details
                    with st.expander("📄 Detailed Growth Information"):
                        st.markdown("### Growth Calculation Details")
                        
                        st.markdown(f"**Current Period:** {latest_period.strftime('%Y-%m-%d')}")
                        st.markdown(f"**Previous Period:** {previous_period.strftime('%Y-%m-%d')}")
                        
                        st.markdown(f"**Current Revenue:** {currency_symbol}{format_number(current_revenue)}" if current_revenue else "**Current Revenue:** N/A")
                        st.markdown(f"**Previous Revenue:** {currency_symbol}{format_number(previous_revenue)}" if previous_revenue else "**Previous Revenue:** N/A")
                        
                        st.markdown(f"**Current Net Income:** {currency_symbol}{format_number(current_net_income)}" if current_net_income else "**Current Net Income:** N/A")
                        st.markdown(f"**Previous Net Income:** {currency_symbol}{format_number(previous_net_income)}" if previous_net_income else "**Previous Net Income:** N/A")
                        
                        st.markdown(f"**Current EPS:** {currency_symbol}{current_eps:.2f}" if current_eps else "**Current EPS:** N/A")
                        st.markdown(f"**Previous EPS:** {currency_symbol}{previous_eps:.2f}" if previous_eps else "**Previous EPS:** N/A")
                else:
                    st.warning("Not enough historical data available to calculate growth metrics.")
            
            # Tab 6: Dividends
            with tabs[5]:
                st.markdown('<h2 class="section-header">Dividend Metrics</h2>', unsafe_allow_html=True)
                
                # Get dividend data
                dividend_yield = info.get("dividendYield", None)
                annual_dividend_rate = info.get("dividendRate", None)
                payout_ratio = info.get("payoutRatio", None)
                
                # Calculate if not available
                if dividend_yield is None and annual_dividend_rate is not None and info.get("currentPrice", None) is not None:
                    dividend_yield = annual_dividend_rate / info.get("currentPrice")
                
                if payout_ratio is None and annual_dividend_rate is not None and info.get("trailingEps", None) is not None:
                    payout_ratio = annual_dividend_rate / info.get("trailingEps")
                
                # Display dividend metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        metric_card("Dividend Yield", f"{dividend_yield*100:.2f}" if dividend_yield else "N/A", suffix="%"),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        metric_card("Annual Dividend", f"{annual_dividend_rate:.2f}" if annual_dividend_rate else "N/A", prefix=currency_symbol),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        metric_card("Payout Ratio", f"{payout_ratio*100:.2f}" if payout_ratio else "N/A", suffix="%"),
                        unsafe_allow_html=True
                    )
                
                # Dividend history
                st.markdown('<h2 class="section-header">Dividend History</h2>', unsafe_allow_html=True)
                
                try:
                    # Get dividend history
                    dividends = ticker.dividends
                    
                    if not dividends.empty:
                        # Create dataframe for chart
                        dividends_df = pd.DataFrame(dividends).reset_index()
                        dividends_df.columns = ['Date', 'Dividend']
                        dividends_df = dividends_df.sort_values('Date')
                        
                        # Create chart
                        fig = px.bar(
                            dividends_df,
                            x='Date',
                            y='Dividend',
                            title='Dividend Payment History',
                            template='plotly_white',
                            color_discrete_sequence=['#1E3A8A']
                        )
                        
                        fig.update_layout(
                            xaxis_title='',
                            yaxis_title=f'Dividend Amount ({currency_symbol})',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate dividend growth
                        if len(dividends_df) > 1:
                            # Group by year and get the sum
                            dividends_df['Year'] = dividends_df['Date'].dt.year
                            annual_dividends = dividends_df.groupby('Year')['Dividend'].sum().reset_index()
                            
                            if len(annual_dividends) > 1:
                                # Calculate year-over-year growth
                                annual_dividends['Previous'] = annual_dividends['Dividend'].shift(1)
                                annual_dividends['Growth'] = (annual_dividends['Dividend'] / annual_dividends['Previous'] - 1) * 100
                                
                                # Display annual dividend growth
                                st.markdown('<h2 class="section-header">Annual Dividend Growth</h2>', unsafe_allow_html=True)
                                
                                fig = px.line(
                                    annual_dividends.dropna(),
                                    x='Year',
                                    y='Growth',
                                    markers=True,
                                    title='Annual Dividend Growth Rate',
                                    template='plotly_white'
                                )
                                
                                fig.update_layout(
                                    xaxis_title='',
                                    yaxis_title='Growth Rate (%)',
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("This company does not pay dividends or has no dividend history available.")
                except Exception as e:
                    st.info("Dividend history data not available.")
                
                # Dividend details
                with st.expander("📄 Detailed Dividend Information"):
                    st.markdown("### Dividend Policy")
                    
                    # This would typically come from company filings or other sources
                    # For demonstration, we'll use a placeholder
                    st.markdown(f"**Dividend Frequency:** {info.get('dividendSchedule', 'Quarterly')}")
                    st.markdown(f"**Ex-Dividend Date:** {info.get('exDividendDate', 'N/A')}")
                    
                    if not dividends.empty:
                        st.markdown("### Recent Dividend Payments")
                        st.dataframe(dividends_df.tail(10).sort_values('Date', ascending=False).style.format({"Dividend": "${:.4f}"}))
    except Exception as e:
        st.error(f"An error occurred while analyzing {ticker_symbol}: {str(e)}")
        st.info("Please check if the ticker symbol is correct and try again.")

# Main app execution
if ticker_symbol:
    analyze_company(ticker_symbol)
else:
    st.info("Enter a ticker symbol to begin analysis.")