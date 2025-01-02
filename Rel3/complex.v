module complex (
    input wire A,
    input wire B,
    input wire C,
    output wire D,
    output wire E
    ) ;

assign D = A & ( B | C ) ;
assign E = B | C;

endmodule