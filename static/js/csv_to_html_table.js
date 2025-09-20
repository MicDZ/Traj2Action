var CsvToHtmlTable = CsvToHtmlTable || {};

CsvToHtmlTable = {
    init: function (options) {
        options = options || {};
        var csv_path = options.csv_path || "";
        var el = options.element || "table-container";
        var allow_download = options.allow_download || false;
        var csv_options = options.csv_options || {};
        var datatables_options = options.datatables_options || {};
        var custom_formatting = options.custom_formatting || [];
        var customTemplates = {};
        $.each(custom_formatting, function (i, v) {
            var colIdx = v[0];
            var func = v[1];
            customTemplates[colIdx] = func;
        });

        var $table = $("<table class='table table-striped table-condensed' id='" + el + "-table'></table>");
        var $containerElement = $("#" + el);
        $containerElement.empty().append($table);

        $.when($.get(csv_path)).then(
            function (data) {
                var csvData = $.csv.toArrays(data, csv_options);
                var $tableHead = $("<thead></thead>");
                var csvHeaderRow = csvData[0];


                var $tableHeadRow2 = $("<tr></tr>");
                

                const api_based_models = ['GPT-4o', 'Gemini-1.5-Pro', 'SenseNova', 'Step-1o', 'Claude-3.5-Sonnet']
    const open_source_models = ['Deepseek-VL2', 'Qwen2.5-VL-72B-Instruct', 'InternVL2-8B', 'InternVL2-26B', 'QVQ-72B-Preview', 'Qwen2-VL-72B-Instruct', 'InternVL2.5-26B-MPO', 'InternVL2.5-78B-MPO']
                for (var headerIdx = 0; headerIdx < csvHeaderRow.length; headerIdx++) {
                    var headerText = csvHeaderRow[headerIdx];
                    // 将headerText里面的空格替换为换行
                    headerText = headerText.replace(/ /g, "\n");
                    console.log(headerText);
                    $tableHeadRow2Cell = $("<th class='tooltip'></th>").text(headerText);
                    $tableHeadRow2.append($tableHeadRow2Cell);
                }
                $tableHeadRow2.css("background-color", "#f5f5f5");
                $tableHead.append($tableHeadRow2);

                $table.append($tableHead);
                var $tableBody = $("<tbody></tbody>");

                for (var rowIdx = 1; rowIdx < csvData.length; rowIdx++) {
                    var $tableBodyRow = $("<tr></tr>");
                    for (var colIdx = 0; colIdx < csvData[rowIdx].length; colIdx++) {
                        var $tableBodyRowTd = $("<td></td>");
                        var cellTemplateFunc = customTemplates[colIdx];
                        if (cellTemplateFunc) {
                            $tableBodyRowTd.html(cellTemplateFunc(csvData[rowIdx][colIdx]));
                        } else {
                            $tableBodyRowTd.text(csvData[rowIdx][colIdx]);
                        }
                        if (colIdx == 0) {
                            // left align
                            $tableBodyRowTd.css("text-align", "left");
                            // click to see more info
                            var cellValue = csvData[rowIdx][colIdx];
                            console.log(cellValue);
                            // if ('Human' in cellValue) 
                            var url = '';
                            if (cellValue.indexOf("Human") > -1) {
                                url =  'https://github.com/MicDZ/MANBench/tree/main/eval/saved_human_results'
                            }
                            else {
                                url = 'https://github.com/MicDZ/MANBench/tree/main/eval/saved_mllms_results';
                            }
                            if (cellValue.indexOf("Random") > -1) {
                                url = '/'
                            }
                            $tableBodyRowTd.html('<a href="' + url + '">' + cellValue + '</a>');
                        }
                        if (colIdx == 1 || colIdx == 0) {
                            $tableBodyRowTd.css("border-right", "1px solid #dbdbdb");
                        }
                        // if API, then set the background color of the row to light red
                        if ($.inArray(csvData[rowIdx][0], api_based_models) > -1) {
                            $tableBodyRow.css("background-color", "#FEFAE3");
                        }
                        // if open source model, then set the background color of the row to light green
                        if ($.inArray(csvData[rowIdx][0], open_source_models) > -1) {
                            $tableBodyRow.css("background-color", "#e8f6ffba");
                        }
                        // if random, light blue
                        if (csvData[rowIdx][0] == "Random") {
                            $tableBodyRow.css("background-color", "#FEF4E4");
                        }
                        // if highest score, bold
                        // check if the value is the highest in the column
                        var isHighest = true;
                        // skip the model Human (Best)
                        
                        for (var i = 1; i < csvData.length; i++) {
                            // skip the model Human (Best)
                            if (csvData[i][0] == "Human (Best)" || csvData[i][0] == "Random" || csvData[i][0] == "Human (Average)") {
                                continue;
                            }
                            if (parseFloat(csvData[rowIdx][colIdx]) < parseFloat(csvData[i][colIdx])) {
                                isHighest = false;
                                break;
                            }
                        }
                        // if the value is higher than Human (Average), underline
                        // get the Human (Average) value
                        var humanAverage = 0;
                        for (var i = 1; i < csvData.length; i++) {
                            if (csvData[i][0] == "Human (Average)" ) {
                                humanAverage = parseFloat(csvData[i][colIdx]);
                                break;
                            }
                        }
                        var underline = false;
                        if (parseFloat(csvData[rowIdx][colIdx]) > humanAverage) {
                            underline = true;
                        }

                        if (csvData[rowIdx][0] == "Human (Best)" || csvData[rowIdx][0] == "Human (Average)" || csvData[rowIdx][0] == "Random") {
                            isHighest = false;
                            underline = false;
                        }
                        if (isHighest) {
                            $tableBodyRowTd.css("font-weight", "bold");
                        }
                        if (underline) {
                            $tableBodyRowTd.css("text-decoration", "underline");
                        }

                        $tableBodyRow.append($tableBodyRowTd);
                        $tableBody.append($tableBodyRow);
                    }
                }
                $table.append($tableBody);
                $table.DataTable(datatables_options);
                if (allow_download) {
                    
                    $containerElement.append("<p class='content has-text-justified'><a class='btn btn-info' href='" + csv_path + "'><i class='glyphicon glyphicon-download'></i> Download as CSV</a></p>");

                }
            });
    }
};