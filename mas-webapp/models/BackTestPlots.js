'use strict';

module.exports = (sequelize, DataTypes) => {
  const BackTestPlots = sequelize.define('BackTestPlots', {
    Name : DataTypes.STRING,
    JSON: DataTypes.TEXT,
    JSONType: DataTypes.TEXT
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'BackTestPlots'
  });

  BackTestPlots.associate = function(models) {
    models.BackTestPlots.belongsTo(models.ModelRunDetail, {
      foreignKey:'ModelId'
    });
  };
  return BackTestPlots;
};
