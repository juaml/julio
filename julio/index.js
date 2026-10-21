$(document).ready(function() {
    $.extend(true, DataTable.ext.classes, {
        container: "dt-container relative m-8",
        length: {
            select: "select w-auto inline-block mr-[0.5em]",
        },
        search: {
            container: "dt-search text-right",
            input: "input w-auto inline-block ml-[0.5em] bag-(--color-base-100) border-(--input-color)",
        },
      	paging: {
            nav: "join",
      		  button: "join-item btn",
      		  active: "btn-active",
            disabled: "btn-disabled",
      	},
        table: "dataTable min-w-full text-sm align-middle whitespace-nowrap",
        layout: {
            row: "flex flex-wrap -mx-2 mt-2 justify-between",
            cell: "md:flex justify-between items-center",
            tableCell: "w-full px-2",
            start: "dt-layout-start me-auto",
            end: "dt-layout-end ms-auto",
		        full: "dt-layout-full",
        },
    });

    $("#feature-table").DataTable({
        "language": {
            "search": "",
            "searchPlaceholder": "Search",
        },
    });
});
