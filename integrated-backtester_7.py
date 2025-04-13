        # Generate parameter range
        window_range = range(window_min, window_max + 1, window_step)
        
        total_combinations = len(list(window_range))
        
        st.write(f"Total combinations to test: {total_combinations}")
        
        if st.button("Run Optimization"):
            if total_combinations > 100:
                st.warning(f"Testing {total_combinations} combinations may take a while. Consider reducing the parameter ranges.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize results list
            results = []
            
            # Prepare data once
            df_pair = prepare_price_data(df, selected_pair)
            
            # Run tests for each combination
            for i, window in enumerate(window_range):
                # Update progress
                progress = (i + 1) / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {i+1}/{total_combinations}: Window={window}")
                
                try:
                    # Create backtest instance
                    backtest = IterativeBacktest(
                        symbol=selected_pair,
                        start=str(df.index.min().date()),
                        end=str(df.index.max().date()),
                        amount=initial_capital,
                        use_spread=use_spread
                    )
                    
                    # Set the prepared data
                    backtest.set_data(df_pair.copy())
                    
                    # Run Contrarian strategy
                    performance = backtest.test_con_strategy(window)
                    
                    # Calculate metrics
                    ann_return = backtest.data["strategy"].mean() * 252
                    ann_risk = backtest.data["strategy"].std() * np.sqrt(252)
                    sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                    
                    # Store results
                    results.append({
                        "Window": window,
                        "performance": performance["final_balance"] / initial_capital,
                        "final_balance": performance["final_balance"],
                        "trades": performance["trades"],
                        "annualized_return": ann_return,
                        "annualized_risk": ann_risk,
                        "sharpe": sharpe
                    })
                except Exception as e:
                    st.error(f"Error with Window={window}: {e}")
                    # Add a placeholder result to keep the loop going
                    results.append({
                        "Window": window,
                        "performance": 1.0,  # Neutral performance (no gain/loss)
                        "final_balance": initial_capital,
                        "trades": 0,
                        "annualized_return": 0,
                        "annualized_risk": 0,
                        "sharpe": 0
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display top performers
            st.subheader("Top 10 Parameter Combinations")
            top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
            top_results["return_pct"] = (top_results["performance"] - 1) * 100  # Convert to percentage
            st.dataframe(top_results[["Window", "return_pct", "final_balance", "trades", "sharpe"]].style.format({
                "return_pct": "{:.2f}%",
                "final_balance": "${:,.2f}",
                "sharpe": "{:.2f}"
            }))
            
            # Allow download of results
            st.markdown(get_table_download_link(results_df, f"{selected_pair}_contrarian_optimization_results.csv"), unsafe_allow_html=True)
            
            # Create line plot for visualization
            st.subheader("Performance by Window Size")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results_df["Window"],
                y=results_df["performance"],
                mode='lines+markers',
                name='Performance Ratio',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title=f"Contrarian Performance by Window Size for {selected_pair}",
                xaxis_title="Window Size",
                yaxis_title="Performance Ratio",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to use best parameters
            best_params = results_df.loc[results_df["performance"].idxmax()]
            
            st.subheader("Best Parameters")
            col1, col2, col3 = st.columns(3)
            col1.metric("Window", int(best_params["Window"]))
            col2.metric("Return", f"{(best_params['performance'] - 1) * 100:.2f}%")
            col3.metric("Trades", int(best_params["trades"]))
            
            st.info(f"💡 Tip: You can use the optimal window size ({int(best_params['Window'])}) in the Contrarian Strategy section.")
    
    elif strategy_type == "Bollinger Parameter Optimization":
        st.header("Bollinger Bands Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Ranges")
        col1, col2 = st.columns(2)
        with col1:
            sma_min = st.number_input("Min SMA", min_value=5, max_value=50, value=10, step=1)
            sma_max = st.number_input("Max SMA", min_value=10, max_value=100, value=50, step=5)
            sma_step = st.number_input("SMA Step", min_value=1, max_value=10, value=5, step=1)
        
        with col2:
            dev_min = st.number_input("Min Std Dev", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            dev_max = st.number_input("Max Std Dev", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
            dev_step = st.number_input("Std Dev Step", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
            
        # Generate parameter ranges
        SMA_range = range(sma_min, sma_max + 1, sma_step)
        DEV_range = np.arange(dev_min, dev_max + dev_step/2, dev_step)  # Add small value to include upper bound
        
        total_combinations = len(list(SMA_range)) * len(DEV_range)
        
        st.write(f"Total combinations to test: {total_combinations}")
        
        if st.button("Run Optimization"):
            if total_combinations > 100:
                st.warning(f"Testing {total_combinations} combinations may take a while. Consider reducing the parameter ranges.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate combinations
            combinations = list(product(SMA_range, DEV_range))
            
            # Initialize results list
            results = []
            
            # Prepare data once
            df_pair = prepare_price_data(df, selected_pair)
            
            # Run tests for each combination
            for i, (sma, dev) in enumerate(combinations):
                # Update progress
                progress = (i + 1) / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {i+1}/{total_combinations}: SMA={sma}, Dev={dev:.1f}")
                
                try:
                    # Create backtest instance
                    backtest = IterativeBacktest(
                        symbol=selected_pair,
                        start=str(df.index.min().date()),
                        end=str(df.index.max().date()),
                        amount=initial_capital,
                        use_spread=use_spread
                    )
                    
                    # Set the prepared data
                    backtest.set_data(df_pair.copy())
                    
                    # Run Bollinger Bands strategy
                    performance = backtest.test_boll_strategy(sma, dev)
                    
                    # Calculate metrics
                    ann_return = backtest.data["strategy"].mean() * 252
                    ann_risk = backtest.data["strategy"].std() * np.sqrt(252)
                    sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                    
                    # Store results
                    results.append({
                        "SMA": sma,
                        "Dev": dev,
                        "performance": performance["final_balance"] / initial_capital,
                        "final_balance": performance["final_balance"],
                        "trades": performance["trades"],
                        "annualized_return": ann_return,
                        "annualized_risk": ann_risk,
                        "sharpe": sharpe
                    })
                except Exception as e:
                    st.error(f"Error with SMA={sma}, Dev={dev:.1f}: {e}")
                    # Add a placeholder result to keep the loop going
                    results.append({
                        "SMA": sma,
                        "Dev": dev,
                        "performance": 1.0,  # Neutral performance (no gain/loss)
                        "final_balance": initial_capital,
                        "trades": 0,
                        "annualized_return": 0,
                        "annualized_risk": 0,
                        "sharpe": 0
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display top performers
            st.subheader("Top 10 Parameter Combinations")
            top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
            top_results["return_pct"] = (top_results["performance"] - 1) * 100  # Convert to percentage
            st.dataframe(top_results[["SMA", "Dev", "return_pct", "final_balance", "trades", "sharpe"]].style.format({
                "Dev": "{:.1f}",
                "return_pct": "{:.2f}%",
                "final_balance": "${:,.2f}",
                "sharpe": "{:.2f}"
            }))
            
            # Allow download of results
            st.markdown(get_table_download_link(results_df, f"{selected_pair}_bollinger_optimization_results.csv"), unsafe_allow_html=True)
            
            # Create heatmap for visualization using Plotly
            st.subheader("Performance Heatmap")
            
            # Convert Dev to string with 1 decimal for clean display
            results_df["Dev_str"] = results_df["Dev"].apply(lambda x: f"{x:.1f}")
            
            pivot_data = results_df.pivot_table(
                index="SMA", 
                columns="Dev_str", 
                values="performance",
                aggfunc='first'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                colorbar=dict(title='Performance Ratio')
            ))
            
            fig.update_layout(
                title=f"Bollinger Parameter Optimization Heatmap for {selected_pair}",
                xaxis_title="Standard Deviation Factor",
                yaxis_title="SMA Period",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to use best parameters
            best_params = results_df.loc[results_df["performance"].idxmax()]
            
            st.subheader("Best Parameters")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("SMA Period", int(best_params["SMA"]))
            col2.metric("Std Dev Factor", f"{best_params['Dev']:.1f}")
            col3.metric("Return", f"{(best_params['performance'] - 1) * 100:.2f}%")
            col4.metric("Trades", int(best_params["trades"]))
            
            st.info(f"💡 Tip: You can use these optimal parameters (SMA={int(best_params['SMA'])}, Dev={best_params['Dev']:.1f}) in the Bollinger Bands Strategy section.")
    
    elif strategy_type == "Cross-Pair Analysis":
        st.header("Cross-Pair Analysis")
        
        if len(comparison_pairs) < 1:
            st.warning("Please select at least one currency pair to analyze.")
        else:
            # Select analysis type
            analysis_type = st.radio(
                "Analysis Type",
                ["Price Comparison", "Returns Comparison", "Strategy Performance"]
            )
            
            if analysis_type == "Price Comparison":
                st.subheader("Price Comparison")
                
                # Normalize prices for comparison
                fig = go.Figure()
                
                for pair in comparison_pairs:
                    # Normalize to starting at 100
                    normalized_price = df[pair] / df[pair].iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=normalized_price, 
                        mode='lines', 
                        name=f'{pair}'
                    ))
                
                fig.update_layout(
                    height=600,
                    title="Normalized Price Comparison (Base=100)",
                    xaxis_title='Date',
                    yaxis_title='Normalized Price'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(comparison_pairs) > 1:
                    st.subheader("Price Correlation Matrix")
                    corr_matrix = df[comparison_pairs].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=500,
                        title="Price Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Returns Comparison":
                st.subheader("Returns Comparison")
                
                # Calculate returns for each pair
                returns_df = pd.DataFrame(index=df.index)
                for pair in comparison_pairs:
                    df_pair = prepare_price_data(df, pair)
                    returns_df[pair] = df_pair["returns"]
                
                returns_df.dropna(inplace=True)
                
                # Calculate cumulative returns
                cum_returns_df = pd.DataFrame(index=returns_df.index)
                for pair in comparison_pairs:
                    cum_returns_df[pair] = returns_df[pair].cumsum().apply(np.exp)
                
                # Plot returns comparison
                fig = go.Figure()
                for pair in comparison_pairs:
                    fig.add_trace(go.Scatter(
                        x=cum_returns_df.index, 
                        y=cum_returns_df[pair], 
                        mode='lines', 
                        name=f'{pair}'
                    ))
                
                fig.update_layout(
                    height=600,
                    title="Cumulative Returns Comparison",
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility comparison
                if len(comparison_pairs) > 0:
                    st.subheader("Volatility Comparison")
                    # Calculate rolling standard deviation (annualized)
                    window = st.slider("Rolling Window (days)", min_value=5, max_value=252, value=30, step=5)
                    
                    vol_df = pd.DataFrame(index=returns_df.index)
                    for pair in comparison_pairs:
                        vol_df[pair] = returns_df[pair].rolling(window).std() * np.sqrt(252)
                    
                    # Plot volatility
                    fig = go.Figure()
                    for pair in comparison_pairs:
                        fig.add_trace(go.Scatter(
                            x=vol_df.index, 
                            y=vol_df[pair], 
                            mode='lines', 
                            name=f'{pair}'
                        ))
                    
                    fig.update_layout(
                        height=500,
                        title=f"Rolling {window}-Day Annualized Volatility",
                        xaxis_title='Date',
                        yaxis_title='Volatility'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Returns correlation matrix
                    st.subheader("Returns Correlation Matrix")
                    returns_corr = returns_df.corr()
                    
                    fig = px.imshow(
                        returns_corr,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=500,
                        title="Returns Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary
                    st.subheader("Returns Statistics")
                    
                    # Create a summary table
                    summary_data = []
                    for pair in comparison_pairs:
                        pair_returns = returns_df[pair]
                        
                        ann_return = pair_returns.mean() * 252
                        ann_vol = pair_returns.std() * np.sqrt(252)
                        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                        
                        summary_data.append({
                            "Pair": pair,
                            "Mean Daily Return": pair_returns.mean(),
                            "Annualized Return": ann_return,
                            "Annualized Volatility": ann_vol,
                            "Sharpe Ratio": sharpe,
                            "Min Return": pair_returns.min(),
                            "Max Return": pair_returns.max()
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df.style.format({
                        "Mean Daily Return": "{:.6f}",
                        "Annualized Return": "{:.2%}",
                        "Annualized Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                        "Min Return": "{:.2%}",
                        "Max Return": "{:.2%}"
                    }))
            
            elif analysis_type == "Strategy Performance":
                st.subheader("Strategy Performance Comparison")
                
                # Strategy selection for comparison
                strategy_selection = st.radio(
                    "Strategy Type",
                    ["SMA Crossover", "Contrarian Strategy", "Bollinger Bands Strategy"]
                )
                
                if strategy_selection == "SMA Crossover":
                    # SMA parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        sma_s = st.slider("Short SMA Period", min_value=5, max_value=100, value=50, step=1)
                    with col2:
                        sma_l = st.slider("Long SMA Period", min_value=50, max_value=300, value=200, step=1)
                    
                    if sma_s >= sma_l:
                        st.error("Short SMA period must be less than Long SMA period")
                    else:
                        try:
                            # Run strategy for each pair
                            results_data = []
                            
                            # Plot for cumulative strategy returns
                            fig = go.Figure()
                            
                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, pair in enumerate(comparison_pairs):
                                # Update progress
                                progress = (i + 1) / len(comparison_pairs)
                                progress_bar.progress(progress)
                                status_text.text(f"Testing pair {i+1}/{len(comparison_pairs)}: {pair}")
                                
                                # Prepare data for this pair
                                df_pair = prepare_price_data(df, pair)
                                
                                # Create backtest instance
                                backtest = IterativeBacktest(
                                    symbol=pair,
                                    start=str(df.index.min().date()),
                                    end=str(df.index.max().date()),
                                    amount=initial_capital,
                                    use_spread=use_spread
                                )
                                
                                # Set the prepared data
                                backtest.set_data(df_pair)
                                
                                # Run SMA strategy
                                performance = backtest.test_sma_strategy(sma_s, sma_l)
                                
                                # Add to plot
                                fig.add_trace(go.Scatter(
                                    x=backtest.data.index, 
                                    y=backtest.data.cstrategy, 
                                    mode='lines', 
                                    name=f'{pair} Strategy'
                                ))
                                
                                # Calculate metrics
                                metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
                                
                                # Store results
                                results_data.append({
                                    "Pair": pair,
                                    "Buy & Hold Return": metrics["Total Return"],
                                    "Strategy Return": metrics["Strategy Return"],
                                    "Outperformance": metrics["Outperformance"],
                                    "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                    "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                    "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                    "Maximum Drawdown": metrics["Maximum Drawdown"],
                                    "Final Balance": performance["final_balance"],
                                    "Total Trades": performance["trades"]
                                })
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Display the cumulative returns plot
                            fig.update_layout(
                                height=600,
                                title=f"SMA({sma_s}, {sma_l}) Strategy Performance by Currency Pair",
                                xaxis_title="Date",
                                yaxis_title="Cumulative Returns"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create results DataFrame and sort by performance
                            results_df = pd.DataFrame(results_data)
                            
                            # Convert percentage strings to floats for sorting
                            # Create a new DataFrame for ranking with numerical values
                            ranking_df = pd.DataFrame(index=results_df.index)
                            for pair in results_df["Pair"]:
                                pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                                ranking_df.loc[pair_row.name, "Pair"] = pair
                                ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                                ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                                ranking_df.loc[pair_row.name, "Final Balance"] = pair_row["Final Balance"]
                                ranking_df.loc[pair_row.name, "Total Trades"] = pair_row["Total Trades"]
                            
                            # Sort by strategy return
                            ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                            
                            # Reorder the original DataFrame based on ranking
                            sorted_results = pd.DataFrame()
                            for idx in ranking_df.index:
                                sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                            
                            # Display the metrics table
                            st.subheader("Performance Metrics by Currency Pair")
                            st.dataframe(sorted_results)
                            
                            # Bar chart comparing key metrics
                            st.subheader("Key Metrics Comparison")
                            
                            # Bar chart for Strategy Returns
                            fig = px.bar(
                                ranking_df,
                                x="Pair",
                                y="Strategy Return",
                                title="Strategy Returns by Currency Pair",
                                color="Strategy Return",
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Bar chart for Sharpe Ratios
                            fig = px.bar(
                                ranking_df,
                                x="Pair",
                                y="Sharpe Ratio",
                                title="Sharpe Ratios by Currency Pair",
                                color="Sharpe Ratio",
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"An error occurred in the SMA strategy comparison: {e}")
                
                elif strategy_selection == "Contrarian Strategy":
                    # Contrarian parameters
                    window = st.slider("Look-back Window", min_value=1, max_value=100, value=10, step=1)
                    
                    try:
                        # Run strategy for each pair
                        results_data = []
                        
                        # Plot for cumulative strategy returns
                        fig = go.Figure()
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, pair in enumerate(comparison_pairs):
                            # Update progress
                            progress = (i + 1) / len(comparison_pairs)
                            progress_bar.progress(progress)
                            status_text.text(f"Testing pair {i+1}/{len(comparison_pairs)}: {pair}")
                            
                            # Prepare data for this pair
                            df_pair = prepare_price_data(df, pair)
                            
                            # Create backtest instance
                            backtest = IterativeBacktest(
                                symbol=pair,
                                start=str(df.index.min().date()),
                                end=str(df.index.max().date()),
                                amount=initial_capital,
                                use_spread=use_spread
                            )
                            
                            # Set the prepared data
                            backtest.set_data(df_pair)
                            
                            # Run Contrarian strategy
                            performance = backtest.test_con_strategy(window)
                            
                            # Add to plot
                            fig.add_trace(go.Scatter(
                                x=backtest.data.index, 
                                y=backtest.data.cstrategy, 
                                mode='lines', 
                                name=f'{pair} Strategy'
                            ))
                            
                            # Calculate metrics
                            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
                            
                            # Store results
                            results_data.append({
                                "Pair": pair,
                                "Buy & Hold Return": metrics["Total Return"],
                                "Strategy Return": metrics["Strategy Return"],
                                "Outperformance": metrics["Outperformance"],
                                "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                "Maximum Drawdown": metrics["Maximum Drawdown"],
                                "Final Balance": performance["final_balance"],
                                "Total Trades": performance["trades"]
                            })
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display the cumulative returns plot
                        fig.update_layout(
                            height=600,
                            title=f"Contrarian Strategy (Window={window}) Performance by Currency Pair",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Returns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create results DataFrame and sort by performance
                        results_df = pd.DataFrame(results_data)
                        
                        # Convert percentage strings to floats for sorting
                        # Create a new DataFrame for ranking with numerical values
                        ranking_df = pd.DataFrame(index=results_df.index)
                        for pair in results_df["Pair"]:
                            pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                            ranking_df.loc[pair_row.name, "Pair"] = pair
                            ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                            ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                            ranking_df.loc[pair_row.name, "Final Balance"] = pair_row["Final Balance"]
                            ranking_df.loc[pair_row.name, "Total Trades"] = pair_row["Total Trades"]
                        
                        # Sort by strategy return
                        ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                        
                        # Reorder the original DataFrame based on ranking
                        sorted_results = pd.DataFrame()
                        for idx in ranking_df.index:
                            sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                        
                        # Display the metrics table
                        st.subheader("Performance Metrics by Currency Pair")
                        st.dataframe(sorted_results)
                        
                        # Bar charts for key metrics
                        st.subheader("Key Metrics Comparison")
                        
                        # Bar chart for Strategy Returns
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Strategy Return",
                            title="Strategy Returns by Currency Pair",
                            color="Strategy Return",
                            color_continuous_scale="RdYlGn"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bar chart for Total Trades
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Total Trades",
                            title="Trading Activity by Currency Pair",
                            color="Total Trades"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred in the Contrarian strategy comparison: {e}")
                
                elif strategy_selection == "Bollinger Bands Strategy":
                    # Bollinger parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        sma = st.slider("SMA Period", min_value=5, max_value=100, value=20, step=1)
                    with col2:
                        dev = st.slider("Standard Deviation Factor", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                    
                    try:
                        # Run strategy for each pair
                        results_data = []
                        
                        # Plot for cumulative strategy returns
                        fig = go.Figure()
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, pair in enumerate(comparison_pairs):
                            # Update progress
                            progress = (i + 1) / len(comparison_pairs)
                            progress_bar.progress(progress)
                            status_text.text(f"Testing pair {i+1}/{len(comparison_pairs)}: {pair}")
                            
                            # Prepare data for this pair
                            df_pair = prepare_price_data(df, pair)
                            
                            # Create backtest instance
                            backtest = IterativeBacktest(
                                symbol=pair,
                                start=str(df.index.min().date()),
                                end=str(df.index.max().date()),
                                amount=initial_capital,
                                use_spread=use_spread
                            )
                            
                            # Set the prepared data
                            backtest.set_data(df_pair)
                            
                            # Run Bollinger Bands strategy
                            performance = backtest.test_boll_strategy(sma, dev)
                            
                            # Add to plot
                            fig.add_trace(go.Scatter(
                                x=backtest.data.index, 
                                y=backtest.data.cstrategy, 
                                mode='lines', 
                                name=f'{pair} Strategy'
                            ))
                            
                            # Calculate metrics
                            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
                            
                            # Store results
                            results_data.append({
                                "Pair": pair,
                                "Buy & Hold Return": metrics["Total Return"],
                                "Strategy Return": metrics["Strategy Return"],
                                "Outperformance": metrics["Outperformance"],
                                "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                "Maximum Drawdown": metrics["Maximum Drawdown"],
                                "Final Balance": performance["final_balance"],
                                "Total Trades": performance["trades"]
                            })
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display the cumulative returns plot
                        fig.update_layout(
                            height=600,
                            title=f"Bollinger Bands Strategy (SMA={sma}, Dev={dev}) Performance by Currency Pair",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Returns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create results DataFrame and sort by performance
                        results_df = pd.DataFrame(results_data)
                        
                        # Convert percentage strings to floats for sorting
                        # Create a new DataFrame for ranking with numerical values
                        ranking_df = pd.DataFrame(index=results_df.index)
                        for pair in results_df["Pair"]:
                            pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                            ranking_df.loc[pair_row.name, "Pair"] = pair
                            ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                            ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                            ranking_df.loc[pair_row.name, "Final Balance"] = pair_row["Final Balance"]
                            ranking_df.loc[pair_row.name, "Total Trades"] = pair_row["Total Trades"]
                        
                        # Sort by strategy return
                        ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                        
                        # Reorder the original DataFrame based on ranking
                        sorted_results = pd.DataFrame()
                        for idx in ranking_df.index:
                            sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                        
                        # Display the metrics table
                        st.subheader("Performance Metrics by Currency Pair")
                        st.dataframe(sorted_results)
                        
                        # Bar charts for key metrics
                        st.subheader("Key Metrics Comparison")
                        
                        # Bar chart for Strategy Returns
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Strategy Return",
                            title="Strategy Returns by Currency Pair",
                            color="Strategy Return",
                            color_continuous_scale="RdYlGn"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bar chart for Total Trades
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Total Trades",
                            title="Trading Activity by Currency Pair",
                            color="Total Trades"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred in the Bollinger Bands strategy comparison: {e}")
else:
    st.warning("Please upload data or use the sample data to proceed with the analysis.")import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import io
import base64
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# Import the iterative backtesting classes
class IterativeBase():
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''

    def __init__(self, symbol, start, end, amount, use_spread = True):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        use_spread: boolean (default = True) 
            whether trading costs (bid-ask spread) are included
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.results = []  # To store trade results
        self.get_data()
    
    def get_data(self):
        ''' Imports the data.
        '''
        # This will be overridden to use the uploaded or sample data
        pass

    def set_data(self, data):
        ''' Sets the data directly from a dataframe.
        '''
        self.data = data.copy()
        if "returns" not in self.data.columns:
            self.data["returns"] = np.log(self.data.price / self.data.price.shift(1))

    def plot_data(self, cols = None):  
        ''' Plots the closing price for the symbol.
        '''
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        spread = 0  # Default in case spread column doesn't exist
        if "spread" in self.data.columns:
            spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a buy order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread/2 # ask price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        
        # Save trade details
        self.results.append({
            "date": self.data.index[bar],
            "price": price,
            "type": "buy",
            "units": units,
            "value": units * price,
            "balance": self.current_balance,
            "position": self.units
        })
        
        #print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a sell order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread/2 # bid price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        
        # Save trade details
        self.results.append({
            "date": self.data.index[bar],
            "price": price,
            "type": "sell",
            "units": units,
            "value": units * price,
            "balance": self.current_balance,
            "position": self.units
        })
        
        #print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        date, price, spread = self.get_values(bar)
        #print(75 * "-")
        #print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        
        # Save closing trade details
        if self.units != 0:
            self.results.append({
                "date": self.data.index[bar],
                "price": price,
                "type": "close",
                "units": self.units,
                "value": self.units * price,
                "balance": self.current_balance,
                "position": 0
            })
        
        #print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        #self.print_current_balance(bar)
        #print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        #print("{} | number of trades executed = {}".format(date, self.trades))
        #print(75 * "-")
        
        # Return performance metrics
        return {
            "final_balance": self.current_balance,
            "performance_pct": perf,
            "trades": self.trades
        }
    
    def get_performance_summary(self):
        '''Returns a summary of the strategy performance'''
        if not self.results:
            return None
        
        trades_df = pd.DataFrame(self.results)
        if trades_df.empty:
            return None
        
        initial = self.initial_balance
        final = self.current_balance
        
        return {
            "initial_balance": initial,
            "final_balance": final,
            "return_pct": (final - initial) / initial * 100,
            "trades": self.trades,
            "trades_df": trades_df
        }


class IterativeBacktest(IterativeBase):
    ''' Class for iterative (event-driven) backtesting of trading strategies.
    '''

    # helper method
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.buy_instrument(bar, units = -self.units) # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount) # go long

    # helper method
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.sell_instrument(bar, units = self.units) # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short

    def test_sma_strategy(self, SMA_S, SMA_L):
        ''' 
        Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
        
        Parameters
        ----------
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)

        # Track positions for plotting
        self.data["position"] = 0

        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()
        
        
    def test_con_strategy(self, window = 1):
        ''' 
        Backtests a simple contrarian strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
        self.data.dropna(inplace = True)
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Contrarian strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.data["rolling_returns"].iloc[bar] <= 0: #signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
            elif self.data["rolling_returns"].iloc[bar] > 0: #signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()
        
        
    def test_boll_strategy(self, SMA, dev):
        ''' 
        Backtests a Bollinger Bands mean-reversion strategy.
        
        Parameters
        ----------
        SMA: int
            moving window in bars (e.g. days) for simple moving average.
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA"] = self.data["price"].rolling(SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
        self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
        self.data.dropna(inplace = True) 
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Bollinger strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.position == 0: # when neutral
                if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
                elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go Short
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            elif self.position == 1: # when long
                if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go short
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
                    else:
                        self.sell_instrument(bar, units = self.units) # go neutral
                        self.position = 0
            elif self.position == -1: # when short
                if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1 # long position
                    else:
                        self.buy_instrument(bar, units = -self.units) # go neutral
                        self.position = 0  
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()

# Set page configuration
st.set_page_config(page_title="Advanced Trading Strategy Backtester", layout="wide")

# App title and description
st.title("Advanced Trading Strategy Backtester")
st.markdown("""
This application allows you to backtest multiple trading strategies:
- Simple Moving Average (SMA) crossover strategy
- Contrarian strategy
- Bollinger Bands strategy

Upload your own data or use the sample data.
""")

# Sidebar for inputs
st.sidebar.header("Settings")

# Data upload or use sample data
data_option = st.sidebar.radio("Data Source", ["Use sample data", "Upload your own data"])

df = None
available_pairs = []

if data_option == "Use sample data":
    try:
        # Create sample multi-currency data
        dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='B')
        np.random.seed(42)  # For reproducibility
        
        sample_data = {'time': dates}
        
        # Simulate multiple currency pairs
        pairs = ['EURUSD', 'GBPUSD', 'EURAUD']
        
        for pair in pairs:
            # Create a price series with some trend and randomness
            start_price = 1.1 if 'EUR' in pair else (1.3 if 'GBP' in pair else 1.5)
            price = start_price
            prices = [price]
            
            # Add a small spread for transaction costs
            spread = 0.0001  # 1 pip spread
            spreads = [spread]
            
            for _ in range(1, len(dates)):
                price *= (1 + np.random.normal(0.0001, 0.005))  # Small daily changes
                prices.append(price)
                spreads.append(spread)
                
            sample_data[pair] = prices
            sample_data[f"{pair}_spread"] = spreads
        
        df = pd.DataFrame(sample_data)
        df.set_index('time', inplace=True)
        
        available_pairs = [col for col in df.columns if not col.endswith('_spread')]
        st.sidebar.success(f"Sample data loaded successfully with {len(available_pairs)} currency pairs")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with currency price data", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["time"], index_col="time")
            
            # Get available currency pairs (all columns except those ending with _spread)
            available_pairs = [col for col in df.columns if not col.endswith('_spread')]
            
            if len(available_pairs) == 0:
                st.sidebar.error("No currency pairs found in the uploaded file")
                df = None
            else:
                st.sidebar.success(f"Data loaded successfully with {len(available_pairs)} currency pairs")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded data: {e}")
            st.sidebar.info("Make sure your CSV has a 'time' column and at least one currency pair column")

# Currency pair selection (only if data is loaded)
selected_pair = None
if df is not None and len(available_pairs) > 0:
    selected_pair = st.sidebar.selectbox("Select Currency Pair", available_pairs, index=0)
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Strategy Type", 
        ["Data Overview", "Buy and Hold", "SMA Crossover", "Contrarian Strategy", "Bollinger Bands Strategy",
         "SMA Parameter Optimization", "Contrarian Parameter Optimization", "Bollinger Parameter Optimization",
         "Cross-Pair Analysis"]
    )
    
    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
    
    # Transaction costs settings
    use_spread = st.sidebar.checkbox("Use Spread for Transaction Costs", value=True)
    
    # For Cross-Pair Analysis, allow selection of multiple pairs for comparison
    if strategy_type == "Cross-Pair Analysis":
        comparison_pairs = st.sidebar.multiselect(
            "Select Pairs to Compare",
            available_pairs,
            default=[available_pairs[0]] if available_pairs else []
        )
    else:
        comparison_pairs = [selected_pair]

# Function to download dataframe as CSV
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to prepare data for strategy testing
def prepare_price_data(data, pair):
    # Create a copy with the selected pair, renamed to 'price'
    df_pair = pd.DataFrame(data[pair].copy()).rename(columns={pair: 'price'})
    
    # Add spread if available
    spread_col = f"{pair}_spread"
    if spread_col in data.columns:
        df_pair["spread"] = data[spread_col]
    else:
        df_pair["spread"] = 0.0001  # Default 1 pip spread
    
    # Calculate returns
    df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
    
    return df_pair

# Function to calculate performance metrics
def calculate_metrics(returns_series, strategy_series=None):
    metrics = {}
    
    # Basic performance
    metrics["Total Return"] = f"{np.exp(returns_series.sum()) - 1:.2%}"
    metrics["Annualized Return"] = f"{returns_series.mean() * 252:.2%}"
    metrics["Annualized Risk"] = f"{returns_series.std() * np.sqrt(252):.2%}"
    # Avoid division by zero for Sharpe ratio
    std = returns_series.std()
    metrics["Sharpe Ratio"] = f"{(returns_series.mean() / std if std > 0 else 0) * np.sqrt(252):.2f}"
    
    # Max Drawdown
    cum_returns = returns_series.cumsum().apply(np.exp)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    metrics["Maximum Drawdown"] = f"{drawdown.min():.2%}"
    
    # Strategy specific metrics
    if strategy_series is not None:
        metrics["Strategy Return"] = f"{np.exp(strategy_series.sum()) - 1:.2%}"
        metrics["Strategy Annualized Return"] = f"{strategy_series.mean() * 252:.2%}"
        metrics["Strategy Annualized Risk"] = f"{strategy_series.std() * np.sqrt(252):.2%}"
        # Avoid division by zero for Strategy Sharpe ratio
        std = strategy_series.std()
        metrics["Strategy Sharpe Ratio"] = f"{(strategy_series.mean() / std if std > 0 else 0) * np.sqrt(252):.2f}"
        metrics["Outperformance"] = f"{np.exp(strategy_series.sum()) - np.exp(returns_series.sum()):.2%}"
    
    return metrics

# Function to get numerical value from metric string for cross-pair analysis
def get_metric_value(metric_str):
    try:
        # Remove % sign and convert to float
        return float(metric_str.strip('%')) / 100
    except:
        # If it's not a percentage, just convert to float
        return float(metric_str)

if df is not None and selected_pair is not None:
    # Display selected currency pair (except for Cross-Pair Analysis)
    if strategy_type != "Cross-Pair Analysis":
        st.header(f"Analysis for {selected_pair}")
    
    if strategy_type == "Data Overview":
        st.header("Data Overview")
        
        # Show basic data information
        st.subheader("Dataset Information")
        st.write(f"Date range: {df.index.min()} to {df.index.max()}")
        st.write(f"Number of observations: {df.shape[0]}")
        st.write(f"Available currency pairs: {', '.join(available_pairs)}")
        
        # Display the head of the dataframe for the selected pair
        st.subheader("Data Preview")
        st.dataframe(df[[selected_pair]].head())
        
        # Plot price data for the selected pair
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[selected_pair], mode='lines', name=f'{selected_pair}'))
        fig.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and plot returns
        df_pair = prepare_price_data(df, selected_pair)
        returns = df_pair["returns"].dropna()
        
        # Using Plotly for returns histogram
        st.subheader("Returns Distribution")
        fig = px.histogram(returns, nbins=100)
        fig.update_layout(
            height=500,
            title=f"{selected_pair} Returns Distribution",
            xaxis_title="Log Returns",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Returns Statistics")
        st.dataframe(returns.describe())
        
    elif strategy_type == "Buy and Hold":
        st.header("Buy and Hold Strategy")
        
        try:
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Calculate buy and hold performance (just buy at start, sell at end)
            backtest.position = 0
            backtest.trades = 0
            backtest.current_balance = initial_capital
            backtest.results = []
            
            # Buy at the beginning
            backtest.go_long(0, amount="all")
            backtest.position = 1
            
            # Close position at the end
            final_bar = len(backtest.data) - 1
            performance = backtest.close_pos(final_bar)
            
            # Calculate metrics
            metrics = calculate_metrics(df_pair.returns.dropna())
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", metrics["Total Return"])
            col2.metric("Annualized Return", metrics["Annualized Return"])
            col3.metric("Annualized Risk", metrics["Annualized Risk"])
            col4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            st.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Display final performance from backtest
            st.subheader("Buy and Hold Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Performance", f"{performance['performance_pct']:.2f}%")
            
            # Plot cumulative returns
            df_pair["creturns"] = df_pair["returns"].cumsum().apply(np.exp)
            
            # Calculate drawdown
            drawdown = (df_pair["creturns"] / df_pair["creturns"].cummax() - 1)
            df_pair["drawdown"] = drawdown
            
            st.subheader("Cumulative Returns")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pair.index, y=df_pair.creturns, mode='lines', name=f'{selected_pair} Buy and Hold'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot drawdown
            st.subheader("Drawdown")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pair.index, 
                y=df_pair.drawdown, 
                mode='lines', 
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ))
            fig.update_layout(
                height=400,
                xaxis_title='Date',
                yaxis_title='Drawdown'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade summary
            if backtest.results:
                st.subheader("Trade Summary")
                trades_df = pd.DataFrame(backtest.results)
                st.dataframe(trades_df)
                st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_buy_and_hold_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_pair[["returns", "creturns", "drawdown"]].tail(10))
            st.markdown(get_table_download_link(df_pair, filename=f"{selected_pair}_buy_and_hold_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Buy and Hold strategy: {e}")
            st.info("Try using a different currency pair or check if the data format is correct.")
            
    elif strategy_type == "SMA Crossover":
        st.header("SMA Crossover Strategy")
        
        # SMA parameters with input validation
        col1, col2 = st.columns(2)
        with col1:
            sma_s = st.slider("Short SMA Period", min_value=5, max_value=100, value=50, step=1)
        with col2:
            sma_l = st.slider("Long SMA Period", min_value=50, max_value=300, value=200, step=1)
            
        if sma_s >= sma_l:
            st.error("Short SMA period must be less than Long SMA period")
        else:
            try:
                # Prepare data for the selected pair
                df_pair = prepare_price_data(df, selected_pair)
                
                # Create backtest instance
                backtest = IterativeBacktest(
                    symbol=selected_pair,
                    start=str(df.index.min().date()),
                    end=str(df.index.max().date()),
                    amount=initial_capital,
                    use_spread=use_spread
                )
                
                # Set the prepared data
                backtest.set_data(df_pair)
                
                # Run SMA strategy
                performance = backtest.test_sma_strategy(sma_s, sma_l)
                
                # Calculate metrics
                metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
                
                # Display metrics
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Buy and Hold Return", metrics["Total Return"])
                col2.metric("Strategy Return", metrics["Strategy Return"])
                col3.metric("Outperformance", metrics["Outperformance"])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy Ann. Return", metrics["Strategy Annualized Return"])
                col2.metric("Strategy Ann. Risk", metrics["Strategy Annualized Risk"])
                col3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe Ratio"])
                col4.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
                
                # Display final performance from backtest
                st.subheader("Backtest Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Initial Balance", f"${initial_capital:,.2f}")
                col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
                col3.metric("Total Trades", f"{performance['trades']}")
                
                # Plot SMAs and price
                st.subheader("Price and Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.price, mode='lines', name=f'{selected_pair}'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA_S, mode='lines', name=f'SMA {sma_s}'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA_L, mode='lines', name=f'SMA {sma_l}'))
                fig.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Price'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot positions
                st.subheader("Trading Positions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=backtest.data.index, 
                    y=backtest.data.position, 
                    mode='lines', 
                    name='Position',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title='Date',
                    yaxis_title='Position (1=Long, -1=Short)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot returns comparison
                st.subheader("Strategy vs Buy and Hold")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.creturns, mode='lines', name='Buy and Hold'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.cstrategy, mode='lines', name='SMA Strategy'))
                fig.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade summary
                if performance['trades_df'] is not None:
                    st.subheader("Trade Summary")
                    trades_df = performance['trades_df']
                    st.dataframe(trades_df)
                    st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_sma_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
                
                # Results table
                st.subheader("Results Table")
                st.dataframe(backtest.data[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
                st.markdown(get_table_download_link(backtest.data, filename=f"{selected_pair}_sma_strategy_results.csv"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred in the SMA Crossover strategy: {e}")
                st.info("Try using different SMA periods or check if the data format is correct.")
    
    elif strategy_type == "Contrarian Strategy":
        st.header("Contrarian Strategy")
        
        # Contrarian parameters
        window = st.slider("Look-back Window", min_value=1, max_value=100, value=10, step=1)
        
        try:
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Run Contrarian strategy
            performance = backtest.test_con_strategy(window)
            
            # Calculate metrics
            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Buy and Hold Return", metrics["Total Return"])
            col2.metric("Strategy Return", metrics["Strategy Return"])
            col3.metric("Outperformance", metrics["Outperformance"])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Ann. Return", metrics["Strategy Annualized Return"])
            col2.metric("Strategy Ann. Risk", metrics["Strategy Annualized Risk"])
            col3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe Ratio"])
            col4.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Display final performance from backtest
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Total Trades", f"{performance['trades']}")
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=backtest.data.index, 
                y=backtest.data.position, 
                mode='lines', 
                name='Position',
                line=dict(color='purple', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Position (1=Long, -1=Short)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot rolling returns (to show what the strategy is reacting to)
            st.subheader("Rolling Returns (Signal)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=backtest.data.index, 
                y=backtest.data["rolling_returns"], 
                mode='lines', 
                name=f'Rolling {window}-Period Returns',
                line=dict(color='orange', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Rolling Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot returns comparison
            st.subheader("Strategy vs Buy and Hold")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.cstrategy, mode='lines', name='Contrarian Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade summary
            if performance['trades_df'] is not None:
                st.subheader("Trade Summary")
                trades_df = performance['trades_df']
                st.dataframe(trades_df)
                st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_contrarian_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(backtest.data[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(backtest.data, filename=f"{selected_pair}_contrarian_strategy_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Contrarian strategy: {e}")
            st.info("Try using a different window parameter or check if the data format is correct.")
    
    elif strategy_type == "Bollinger Bands Strategy":
        st.header("Bollinger Bands Strategy")
        
        # Bollinger parameters
        col1, col2 = st.columns(2)
        with col1:
            sma = st.slider("SMA Period", min_value=5, max_value=100, value=20, step=1)
        with col2:
            dev = st.slider("Standard Deviation Factor", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            
        try:
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Run Bollinger Bands strategy
            performance = backtest.test_boll_strategy(sma, dev)
            
            # Calculate metrics
            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Buy and Hold Return", metrics["Total Return"])
            col2.metric("Strategy Return", metrics["Strategy Return"])
            col3.metric("Outperformance", metrics["Outperformance"])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Ann. Return", metrics["Strategy Annualized Return"])
            col2.metric("Strategy Ann. Risk", metrics["Strategy Annualized Risk"])
            col3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe Ratio"])
            col4.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Display final performance from backtest
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Total Trades", f"{performance['trades']}")
            
            # Plot Bollinger Bands and price
            st.subheader("Price and Bollinger Bands")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.price, mode='lines', name=f'{selected_pair}'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA, mode='lines', name=f'SMA {sma}'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.Upper, mode='lines', name='Upper Band'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.Lower, mode='lines', name='Lower Band'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Price'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=backtest.data.index, 
                y=backtest.data.position, 
                mode='lines', 
                name='Position',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Position (1=Long, -1=Short)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot returns comparison
            st.subheader("Strategy vs Buy and Hold")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.cstrategy, mode='lines', name='Bollinger Bands Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade summary
            if performance['trades_df'] is not None:
                st.subheader("Trade Summary")
                trades_df = performance['trades_df']
                st.dataframe(trades_df)
                st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_bollinger_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(backtest.data[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(backtest.data, filename=f"{selected_pair}_bollinger_strategy_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Bollinger Bands strategy: {e}")
            st.info("Try using different parameters or check if the data format is correct.")
            
    elif strategy_type == "SMA Parameter Optimization":
        st.header("SMA Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Ranges")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Short SMA Settings**")
            sma_s_min = st.number_input("Min Short SMA", min_value=5, max_value=50, value=10, step=1)
            sma_s_max = st.number_input("Max Short SMA", min_value=10, max_value=100, value=50, step=1)
            sma_s_step = st.number_input("Short SMA Step", min_value=1, max_value=10, value=5, step=1)
        
        with col2:
            st.markdown("**Long SMA Settings**")
            sma_l_min = st.number_input("Min Long SMA", min_value=50, max_value=150, value=100, step=5)
            sma_l_max = st.number_input("Max Long SMA", min_value=100, max_value=300, value=200, step=5)
            sma_l_step = st.number_input("Long SMA Step", min_value=5, max_value=20, value=10, step=5)
            
        if sma_s_max >= sma_l_min:
            st.error("Max Short SMA must be less than Min Long SMA")
        else:
            SMA_S_range = range(sma_s_min, sma_s_max + 1, sma_s_step)
            SMA_L_range = range(sma_l_min, sma_l_max + 1, sma_l_step)
            
            total_combinations = len(list(SMA_S_range)) * len(list(SMA_L_range))
            
            st.write(f"Total combinations to test: {total_combinations}")
            
            if st.button("Run Optimization"):
                if total_combinations > 100:
                    st.warning(f"Testing {total_combinations} combinations may take a while. Consider reducing the parameter ranges.")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate combinations
                combinations = list(product(SMA_S_range, SMA_L_range))
                
                # Initialize results list
                results = []
                
                # Prepare data once
                df_pair = prepare_price_data(df, selected_pair)
                
                # Run tests for each combination
                for i, (s_sma, l_sma) in enumerate(combinations):
                    # Update progress
                    progress = (i + 1) / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing combination {i+1}/{total_combinations}: SMA_S={s_sma}, SMA_L={l_sma}")
                    
                    try:
                        # Create backtest instance
                        backtest = IterativeBacktest(
                            symbol=selected_pair,
                            start=str(df.index.min().date()),
                            end=str(df.index.max().date()),
                            amount=initial_capital,
                            use_spread=use_spread
                        )
                        
                        # Set the prepared data
                        backtest.set_data(df_pair.copy())
                        
                        # Run SMA strategy
                        performance = backtest.test_sma_strategy(s_sma, l_sma)
                        
                        # Calculate metrics
                        ann_return = backtest.data["strategy"].mean() * 252
                        ann_risk = backtest.data["strategy"].std() * np.sqrt(252)
                        sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                        
                        # Store results
                        results.append({
                            "SMA_S": s_sma,
                            "SMA_L": l_sma,
                            "performance": performance["final_balance"] / initial_capital,
                            "final_balance": performance["final_balance"],
                            "trades": performance["trades"],
                            "annualized_return": ann_return,
                            "annualized_risk": ann_risk,
                            "sharpe": sharpe
                        })
                    except Exception as e:
                        st.error(f"Error with SMA_S={s_sma}, SMA_L={l_sma}: {e}")
                        # Add a placeholder result to keep the loop going
                        results.append({
                            "SMA_S": s_sma,
                            "SMA_L": l_sma,
                            "performance": 1.0,  # Neutral performance (no gain/loss)
                            "final_balance": initial_capital,
                            "trades": 0,
                            "annualized_return": 0,
                            "annualized_risk": 0,
                            "sharpe": 0
                        })
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display top performers
                st.subheader("Top 10 Parameter Combinations")
                top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
                top_results["return_pct"] = (top_results["performance"] - 1) * 100  # Convert to percentage
                st.dataframe(top_results[["SMA_S", "SMA_L", "return_pct", "final_balance", "trades", "sharpe"]].style.format({
                    "return_pct": "{:.2f}%",
                    "final_balance": "${:,.2f}",
                    "sharpe": "{:.2f}"
                }))
                
                # Allow download of results
                st.markdown(get_table_download_link(results_df, f"{selected_pair}_sma_optimization_results.csv"), unsafe_allow_html=True)
                
                # Create heatmap for visualization using Plotly
                st.subheader("Performance Heatmap")
                
                pivot_data = results_df.pivot_table(
                    index="SMA_S", 
                    columns="SMA_L", 
                    values="performance",
                    aggfunc='first'
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Performance Ratio')
                ))
                
                fig.update_layout(
                    title=f"SMA Parameter Optimization Heatmap for {selected_pair}",
                    xaxis_title="Long SMA (SMA_L)",
                    yaxis_title="Short SMA (SMA_S)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to use best parameters
                best_params = results_df.loc[results_df["performance"].idxmax()]
                
                st.subheader("Best Parameters")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Short SMA", int(best_params["SMA_S"]))
                col2.metric("Long SMA", int(best_params["SMA_L"]))
                col3.metric("Return", f"{(best_params['performance'] - 1) * 100:.2f}%")
                col4.metric("Trades", int(best_params["trades"]))
                
                st.info(f"💡 Tip: You can use these optimal parameters ({int(best_params['SMA_S'])}, {int(best_params['SMA_L'])}) in the SMA Crossover strategy section.")
    
    elif strategy_type == "Contrarian Parameter Optimization":
        st.header("Contrarian Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Range")
        window_min = st.number_input("Min Window", min_value=1, max_value=50, value=1, step=1)
        window_max = st.number_input("Max Window", min_value=5, max_value=100, value=30, step=1)
        window_step = st.number_input("Window Step", min_value=1, max_value=5, value=1, step=1)
        
        # Generate parameter range
        window_range = range(window_min, window_max + 1, window_step)
        
        total_combinations = len(list(window_range))